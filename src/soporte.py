"""Preparación, cache, entrenamiento, inferencia y métricas del LAB2."""

import copy
import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from asda import aplicar_bqa, ssa_filtros


def estadisticas_train(dataset, limite=None, tam_lote=512):
    archivo = dataset._open()
    X_h5, y_h5 = archivo["X"], archivo["y"]
    ids = dataset.ids if limite is None else dataset.ids[:limite]
    n_canales = X_h5.shape[2]

    n_x = n_y = 0
    suma_x = np.zeros(n_canales, dtype=np.float64)
    suma2_x = np.zeros(n_canales, dtype=np.float64)
    suma_y = suma2_y = 0.0

    for inicio in range(0, len(ids), tam_lote):
        filas = ids[inicio:inicio + tam_lote]
        X = X_h5[filas].astype(np.float64).reshape(-1, n_canales)
        y = y_h5[filas].astype(np.float64).ravel()

        suma_x += X.sum(axis=0)
        suma2_x += (X ** 2).sum(axis=0)
        n_x += X.shape[0]
        suma_y += y.sum()
        suma2_y += (y ** 2).sum()
        n_y += y.size

    media_x = suma_x / n_x
    std_x = np.sqrt(np.maximum(suma2_x / n_x - media_x ** 2, 1e-12))
    media_y = suma_y / n_y
    std_y = np.sqrt(max(suma2_y / n_y - media_y ** 2, 1e-12))

    return media_x.astype(np.float32), std_x.astype(np.float32), float(media_y), float(std_y)


def seleccionar_indices(total, max_muestras, semilla=42):
    if max_muestras is None:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(semilla)
    return np.sort(rng.choice(total, max_muestras, replace=False).astype(np.int64))


def nombre_cache(cache_dir, split):
    return cache_dir / f"precalculo_{split}_full.h5"


def _cache_valida(ruta, indices, epsilon, theta, media_caudal, std_caudal,
                   forzar_recalculo=False):
    if forzar_recalculo or not ruta.exists():
        return False

    try:
        with h5py.File(ruta, "r") as f:
            requeridos = {"trend", "fluctuation", "trend_bqa", "fluctuation_bqa", "indices"}
            return (
                requeridos.issubset(f.keys())
                and f["trend"].shape == (len(indices), 336)
                and np.array_equal(f["indices"][:], indices)
                and np.isclose(f.attrs["epsilon"], epsilon)
                and np.isclose(f.attrs["theta"], theta)
                and np.isclose(f.attrs["media_caudal"], float(media_caudal))
                and np.isclose(f.attrs["std_caudal"], float(std_caudal))
            )
    except Exception:
        return False


def precalcular_asda(dataset, indices, ruta, *, device, media_caudal, std_caudal,
                     epsilon, theta, batch_size_cache=32, forzar_recalculo=False):
    """Precalcula SSA+BQA utilizando únicamente las 336 horas históricas."""
    if _cache_valida(
        ruta, indices, epsilon, theta, media_caudal, std_caudal, forzar_recalculo
    ):
        print(f"Cache existente y válido: {ruta}")
        return 0.0

    if ruta.exists():
        ruta.unlink()

    print(f"Creando cache ASDA: {ruta}")
    inicio_total = time.perf_counter()
    X_h5 = dataset._open()["X"]

    with h5py.File(ruta, "w") as cache:
        datasets = {
            nombre: cache.create_dataset(nombre, shape=(len(indices), 336), dtype="float32")
            for nombre in ["trend", "fluctuation", "trend_bqa", "fluctuation_bqa"]
        }
        cache.create_dataset("indices", data=indices, dtype="int64")
        cache.attrs["epsilon"] = epsilon
        cache.attrs["theta"] = theta
        cache.attrs["media_caudal"] = float(media_caudal)
        cache.attrs["std_caudal"] = float(std_caudal)

        for inicio in range(0, len(indices), batch_size_cache):
            fin = min(inicio + batch_size_cache, len(indices))
            filas_h5 = dataset.ids[indices[inicio:fin]]
            caudal = X_h5[filas_h5, :, 11].astype(np.float32)
            caudal = torch.from_numpy((caudal - media_caudal) / std_caudal).to(device)

            with torch.no_grad():
                trend, fluct = ssa_filtros(caudal, epsilon=epsilon, theta=theta)
                trend_bqa = aplicar_bqa(trend, caudal)
                fluct_bqa = aplicar_bqa(fluct, caudal)

            datasets["trend"][inicio:fin] = trend.cpu().numpy()
            datasets["fluctuation"][inicio:fin] = fluct.cpu().numpy()
            datasets["trend_bqa"][inicio:fin] = trend_bqa.cpu().numpy()
            datasets["fluctuation_bqa"][inicio:fin] = fluct_bqa.cpu().numpy()

            if inicio == 0 or fin == len(indices) or (inicio // batch_size_cache) % 20 == 0:
                print(f"  {fin:>7}/{len(indices)} muestras")

    segundos = time.perf_counter() - inicio_total
    print(f"Cache terminado en {segundos/60:.3f} min")
    return segundos


class DatosNormalizados(Dataset):
    def __init__(self, dataset, indices, normalizar_X, normalizar_y, con_y=True):
        self.dataset, self.indices = dataset, indices
        self.normalizar_X, self.normalizar_y = normalizar_X, normalizar_y
        self.con_y = con_y

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, posicion):
        muestra = self.dataset[int(self.indices[posicion])]
        X = torch.from_numpy(self.normalizar_X(muestra["X"]))
        if self.con_y:
            return X, torch.from_numpy(self.normalizar_y(muestra["y"]))
        return X, muestra["Id"]


class DatosConPrecalculo(Dataset):
    def __init__(self, dataset, indices, ruta_cache, usar_bqa, normalizar_X,
                 normalizar_y, con_y=True):
        self.dataset, self.indices = dataset, indices
        self.ruta_cache, self.usar_bqa = ruta_cache, usar_bqa
        self.normalizar_X, self.normalizar_y = normalizar_X, normalizar_y
        self.con_y, self._cache = con_y, None

    def __len__(self):
        return len(self.indices)

    def _abrir_cache(self):
        if self._cache is None:
            self._cache = h5py.File(self.ruta_cache, "r")
        return self._cache

    def __getitem__(self, posicion):
        muestra = self.dataset[int(self.indices[posicion])]
        cache = self._abrir_cache()
        X = torch.from_numpy(self.normalizar_X(muestra["X"]))
        prefijo = "trend_bqa" if self.usar_bqa else "trend"
        prefijo_f = "fluctuation_bqa" if self.usar_bqa else "fluctuation"
        trend = torch.from_numpy(cache[prefijo][posicion])
        fluct = torch.from_numpy(cache[prefijo_f][posicion])

        if self.con_y:
            return X, trend, fluct, torch.from_numpy(self.normalizar_y(muestra["y"]))
        return X, trend, fluct, muestra["Id"]


def rmse(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_pred - y_real) ** 2)))


def mae(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    return float(np.mean(np.abs(y_pred - y_real)))


def nse(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    denominador = np.sum((y_real - y_real.mean()) ** 2)
    return float(1 - np.sum((y_pred - y_real) ** 2) / max(denominador, 1e-12))


def kge(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real).ravel(), np.asarray(y_pred).ravel()
    r = np.corrcoef(y_real, y_pred)[0, 1]
    media_real, media_pred = y_real.mean(), y_pred.mean()
    beta = media_pred / max(abs(media_real), 1e-12)
    cv_real = y_real.std() / max(abs(media_real), 1e-12)
    cv_pred = y_pred.std() / max(abs(media_pred), 1e-12)
    gamma = cv_pred / max(abs(cv_real), 1e-12)
    return float(1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2))


def pbias(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real).ravel(), np.asarray(y_pred).ravel()
    mascara = np.abs(y_real) > 1e-8
    if not mascara.any():
        return np.nan
    return float(100 * np.mean((y_pred[mascara] - y_real[mascara]) / y_real[mascara]))


def _ejecutar_lote(modelo, lote, device):
    if len(lote) == 2:
        X, y = lote
        return modelo(X.to(device)), y.to(device)
    X, trend, fluct, y = lote
    return modelo(X.to(device), trend.to(device), fluct.to(device)), y.to(device)


def predecir(modelo, loader, device, desnormalizar_y):
    modelo.eval()
    reales, predicciones = [], []
    with torch.no_grad():
        for lote in loader:
            pred, y = _ejecutar_lote(modelo, lote, device)
            predicciones.append(desnormalizar_y(pred.cpu().numpy()))
            reales.append(desnormalizar_y(y.cpu().numpy()))
    return np.concatenate(reales), np.concatenate(predicciones)


def evaluar(modelo, loader, device, desnormalizar_y):
    y_real, y_pred = predecir(modelo, loader, device, desnormalizar_y)
    return {
        "RMSE": rmse(y_real, y_pred), "MAE": mae(y_real, y_pred),
        "NSE": nse(y_real, y_pred), "KGE": kge(y_real, y_pred),
        "PBIAS": pbias(y_real, y_pred),
    }


def entrenar(modelo, train_loader, val_loader, max_epochs, nombre, perdida, *,
             device, desnormalizar_y, patience=2, min_epochs=2):
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    mejor_rmse, mejor_estado, mejor_epoca = np.inf, None, 0
    epocas_sin_mejora, historial = 0, []
    tiempo_train_total = tiempo_val_total = 0.0
    detenido_antes = False

    for epoca in range(1, max_epochs + 1):
        inicio_train = time.perf_counter()
        modelo.train()
        perdidas = []

        for lote in train_loader:
            optimizador.zero_grad()
            pred, y = _ejecutar_lote(modelo, lote, device)
            valor = perdida(pred, y)
            valor.backward()
            optimizador.step()
            perdidas.append(valor.item())

        tiempo_train = time.perf_counter() - inicio_train
        tiempo_train_total += tiempo_train
        inicio_val = time.perf_counter()
        metricas = evaluar(modelo, val_loader, device, desnormalizar_y)
        tiempo_val = time.perf_counter() - inicio_val
        tiempo_val_total += tiempo_val

        rmse_actual = metricas["RMSE"]
        mejora = rmse_actual < mejor_rmse
        if mejora:
            mejor_rmse, mejor_epoca = rmse_actual, epoca
            mejor_estado = copy.deepcopy(modelo.state_dict())
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1

        historial.append({
            "epoch": epoca, "loss_train": float(np.mean(perdidas)),
            "train_min": tiempo_train / 60, "validation_min": tiempo_val / 60,
            "mejora": mejora, **metricas,
        })
        estado = "✓ mejor" if mejora else f"sin mejora {epocas_sin_mejora}/{patience}"
        print(
            f"{nombre} | época {epoca}/{max_epochs} | loss={np.mean(perdidas):.4f} | "
            f"RMSE val={rmse_actual:.5f} mm/h | train={tiempo_train/60:.2f} min | "
            f"val={tiempo_val/60:.2f} min | {estado}"
        )

        if epoca >= min_epochs and epocas_sin_mejora >= patience:
            detenido_antes = True
            print(f"EARLY STOPPING → {nombre}: mejor época={mejor_epoca}, RMSE={mejor_rmse:.5f}")
            break

    modelo.load_state_dict(mejor_estado)
    metricas_finales = evaluar(modelo, val_loader, device, desnormalizar_y)
    resumen = {
        "modelo": nombre, "epocas_ejecutadas": len(historial),
        "mejor_epoca": mejor_epoca, "detenido_antes": detenido_antes,
        "mejor_rmse": mejor_rmse, "train_min": tiempo_train_total / 60,
        "validation_min": tiempo_val_total / 60,
    }
    return historial, metricas_finales, resumen


def predecir_test(modelo, loader, device, desnormalizar_y):
    """Predicción de test para baseline o ASDA según la estructura del lote."""
    modelo.eval()
    ids, predicciones = [], []
    with torch.no_grad():
        for lote in loader:
            if len(lote) == 2:
                X, id_lote = lote
                pred = modelo(X.to(device))
            else:
                X, trend, fluct, id_lote = lote
                pred = modelo(X.to(device), trend.to(device), fluct.to(device))
            ids.append(np.asarray(id_lote))
            predicciones.append(desnormalizar_y(pred.cpu().numpy()))
    return np.concatenate(ids), np.concatenate(predicciones)
