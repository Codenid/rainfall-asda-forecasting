"""Componentes del modelo ASDA adaptado al Laboratorio 2."""

import math
import torch
from torch import nn


def promedio_antidiagonales(matriz):
    """Diagonal averaging: [B,L,K] -> [B,N]."""
    batch, filas, columnas = matriz.shape
    serie = matriz.new_zeros(batch, filas + columnas - 1)
    conteos = matriz.new_zeros(filas + columnas - 1)

    for columna in range(columnas):
        serie[:, columna:columna + filas] += matriz[:, :, columna]
        conteos[columna:columna + filas] += 1

    return serie / conteos


def ssa_filtros(caudal, epsilon=0.05, theta=0.01, incluir_ruido=False):
    """Adaptive SSA sobre la ventana histórica de caudal."""
    L = caudal.shape[1] // 2
    trayectoria = caudal.unfold(1, L, 1).transpose(1, 2)
    U, S, Vh = torch.linalg.svd(trayectoria, full_matrices=False)
    contribucion = S.square() / S.square().sum(dim=1, keepdim=True)

    mascara_trend = contribucion >= epsilon
    mascara_fluct = (contribucion >= theta) & (contribucion < epsilon)

    def reconstruir(mascara):
        componentes = (U * (S * mascara)[:, None, :]) @ Vh
        return promedio_antidiagonales(componentes)

    trend = reconstruir(mascara_trend)
    fluct = reconstruir(mascara_fluct)

    if not incluir_ruido:
        return trend, fluct

    noise = reconstruir(contribucion < theta)
    return trend, fluct, noise


def aplicar_bqa(componente, caudal):
    """Back-Query Attention del paper, sin parámetros entrenables."""
    scores = componente.unsqueeze(2) * caudal.unsqueeze(1)
    A = torch.softmax(scores, dim=-1)
    atendido = torch.bmm(A.transpose(1, 2), componente.unsqueeze(-1)).squeeze(-1)
    return atendido + componente


class BaselineLSTM(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(12, hidden_size, batch_first=True)
        self.salida = nn.Linear(hidden_size, 48)

    def forward(self, X, trend=None, fluct=None):
        _, (h, _) = self.lstm(X)
        return self.salida(h[-1])


class TSAAdaptado(nn.Module):
    def __init__(self, hidden_size, history=336, horizon=48):
        super().__init__()
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.runoff_residual = nn.Linear(1, hidden_size)
        self.proyeccion_futuro = nn.Linear(history, horizon)
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        self.salida = nn.Linear(hidden_size, 1)

    def forward(self, feat_trend, feat_fluct, caudal):
        X_fusion = self.fusion(torch.cat([feat_trend, feat_fluct], dim=-1))
        X_fusion = X_fusion + self.runoff_residual(caudal.unsqueeze(-1))

        media = X_fusion.mean(dim=1, keepdim=True)
        std = X_fusion.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
        X_norm = (X_fusion - media) / std

        futuro = self.proyeccion_futuro(X_norm.transpose(1, 2)).transpose(1, 2)
        Q, K, V = self.Wq(futuro), self.Wk(futuro), self.Wv(futuro)
        A = torch.softmax(
            torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(Q.shape[-1]),
            dim=-1,
        )
        Z = torch.matmul(A, V)
        Z = Z * std + media
        return self.salida(Z).squeeze(-1)


class ASDA(nn.Module):
    def __init__(self, hidden_size=64, con_atencion=True, bqa_precalculado=False):
        super().__init__()
        self.con_atencion = con_atencion
        self.bqa_precalculado = bqa_precalculado
        self.lstm_trend = nn.LSTM(12, hidden_size, batch_first=True)
        self.lstm_fluct = nn.LSTM(12, hidden_size, batch_first=True)

        if con_atencion:
            self.tsa = TSAAdaptado(hidden_size)
        else:
            self.salida_simple = nn.Linear(hidden_size, 48)

    def forward(self, X, trend, fluct):
        meteo = X[:, :, :11]
        caudal = X[:, :, 11]

        if self.con_atencion and not self.bqa_precalculado:
            trend = aplicar_bqa(trend, caudal)
            fluct = aplicar_bqa(fluct, caudal)

        entrada_trend = torch.cat([meteo, trend.unsqueeze(-1)], dim=2)
        entrada_fluct = torch.cat([meteo, fluct.unsqueeze(-1)], dim=2)
        feat_trend, _ = self.lstm_trend(entrada_trend)
        feat_fluct, _ = self.lstm_fluct(entrada_fluct)

        if self.con_atencion:
            return self.tsa(feat_trend, feat_fluct, caudal)

        fusion = (feat_trend[:, -1] + feat_fluct[:, -1]) / 2
        return self.salida_simple(fusion)


def rmse_torch(y_pred, y_real):
    return torch.sqrt(torch.mean((y_pred - y_real) ** 2))


def soft_dtw(y_pred, y_real, gamma=0.1):
    """Soft-DTW wavefront: misma recurrencia, antidiagonales paralelas."""
    distancia = (y_pred.unsqueeze(2) - y_real.unsqueeze(1)).square()
    batch, pasos, _ = distancia.shape
    R = torch.full(
        (batch, pasos + 1, pasos + 1), float("inf"),
        device=y_pred.device, dtype=y_pred.dtype,
    )
    R[:, 0, 0] = 0.0

    for suma in range(2, 2 * pasos + 1):
        i = torch.arange(max(1, suma - pasos), min(pasos, suma - 1) + 1,
                         device=y_pred.device)
        j = suma - i
        anteriores = torch.stack(
            [R[:, i - 1, j], R[:, i, j - 1], R[:, i - 1, j - 1]], dim=-1
        )
        minimo_suave = -gamma * torch.logsumexp(-anteriores / gamma, dim=-1)
        R[:, i, j] = distancia[:, i - 1, j - 1] + minimo_suave

    return R[:, pasos, pasos].mean()


def soft_dtw_referencia(y_pred, y_real, gamma=0.1):
    """Versión secuencial usada para validar la equivalencia de wavefront."""
    distancia = (y_pred.unsqueeze(2) - y_real.unsqueeze(1)).square()
    batch, pasos, _ = distancia.shape
    R = torch.full(
        (batch, pasos + 1, pasos + 1), float("inf"),
        device=y_pred.device, dtype=y_pred.dtype,
    )
    R[:, 0, 0] = 0.0

    for i in range(1, pasos + 1):
        for j in range(1, pasos + 1):
            anteriores = torch.stack(
                [R[:, i - 1, j], R[:, i, j - 1], R[:, i - 1, j - 1]], dim=1
            )
            minimo_suave = -gamma * torch.logsumexp(-anteriores / gamma, dim=1)
            R[:, i, j] = distancia[:, i - 1, j - 1] + minimo_suave

    return R[:, pasos, pasos].mean()


def rsd_loss(y_pred, y_real, lambda_rmse=0.817, lambda_soft_dtw=0.23, gamma=0.1):
    """RSD Loss = lambda1*RMSE + lambda2*soft-DTW."""
    return (
        lambda_rmse * rmse_torch(y_pred, y_real)
        + lambda_soft_dtw * soft_dtw(y_pred, y_real, gamma=gamma)
    )
