# LABORATORIO 2 — Pronóstico de caudal con ASDA

Implementación académica adaptada del enfoque ASDA para pronosticar las siguientes **48 horas de caudal específico** a partir de **336 horas históricas** y 11 variables meteorológicas.

## Estructura del proyecto

```text
LAB2_ASDA_ENTREGA_FINAL/
├── src/
│   ├── asda.py                     # Adaptive SSA, BQA, modelos, TSA y RSD Loss
│   ├── soporte.py                  # Cache, datasets, entrenamiento, métricas e inferencia
│   └── leer_datos.py               # Lectura bajo demanda de los HDF5
├── notebooks/
│   └── Lab2.ipynb                  # Paper, configuración, experimentos y análisis
├── data/                           # Metadata versionada; datasets locales ignorados
│   ├── metadata.json
│   ├── train.h5
│   ├── test.h5
│   └── test_targets.csv
├── artifacts/                      # Caché y checkpoints generados; ignorados por Git
│   ├── cache/
│   └── checkpoints/
├── outputs/
│   └── predicciones_lab2_ASDA_FINAL.csv
└── README.md
```

El notebook es el punto central del laboratorio. Mantiene visibles las decisiones metodológicas, las adaptaciones respecto al paper, los parámetros del experimento, las validaciones y los resultados. Los módulos `.py` encapsulan únicamente implementación repetitiva o detalle algorítmico que no necesita ocupar el flujo principal del notebook.

## Diseño experimental

- **E1 — Baseline LSTM:** `X [336,12] → LSTM → Linear(48)`.
- **E2 — ASDA:** `Adaptive SSA → BQA → Dual LSTM → TSA → RSD Loss`.
- **E3 — Ablación sin Dual Attention:** conserva SSA, Dual LSTM y RSD Loss; elimina BQA y TSA.

La selección final del modelo se realiza exclusivamente con **validation** y utilizando **RMSE** como criterio principal para early stopping.

## Datos requeridos

Ubicar en `data/` los archivos entregados para el laboratorio:

```text
metadata.json
train.h5
test.h5
test_targets.csv
```

El lector HDF5 trabaja bajo demanda y utiliza `split=0` para train y `split=1` para validation cuando el dataset contiene la columna `split`.
`test_targets.csv` también se guarda en `data/` y se utiliza únicamente en la evaluación final de test.

## Flujo de ejecución

```text
Carga de datos
    ↓
Normalización calculada solo con train
    ↓
Adaptive SSA sobre las 336 horas históricas
    ↓
Precálculo SSA + BQA sin utilizar el target futuro
    ↓
E1 / E2 / E3
    ↓
Early stopping por RMSE de validation
    ↓
RMSE / MAE / NSE / KGE / PBIAS
    ↓
Análisis por horizonte de 48 horas
    ↓
Selección del modelo
    ↓
Predicción de test
```

## Evidencia de la corrida final

La ejecución registrada en el notebook utiliza el dataset completo:

- Train: **254,000** muestras.
- Validation: **18,142** muestras.
- Test: **27,983** muestras.
- E1 Baseline LSTM: `RMSE = 0.10373`.
- E2 ASDA: `RMSE = 0.13549`.
- E3 ASDA sin atención: `RMSE = 0.11498`.
- Modelo seleccionado: **E1 Baseline LSTM**.
- Tiempo total registrado: **4.84 horas**.

El notebook conserva la validación de reconstrucción SSA, el control de data leakage, la equivalencia de soft-DTW wavefront, la comparación E1/E2/E3, el análisis de errores, las limitaciones respecto al paper y la trazabilidad de los componentes solicitados en el laboratorio. Las predicciones finales se escriben en `outputs/`; la caché y los checkpoints se generan en `artifacts/`.

## Dependencias principales

Python 3.11, PyTorch, NumPy, pandas, matplotlib y h5py.
