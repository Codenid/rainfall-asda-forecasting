# Pronóstico de caudal a 48 horas: adaptación de ASDA

En este experimento, una LSTM base obtuvo menor error de validación que la adaptación de ASDA. **¿Por qué una arquitectura más compleja no mejoró el pronóstico de caudal para las siguientes 48 horas?** Este trabajo compara ambos enfoques con datos de varias cuencas y analiza tanto el desempeño promedio como la respuesta ante eventos extremos.

## Objetivo y fundamento

El objetivo fue pronosticar el caudal específico de las próximas **48 horas** usando **336 horas de historia**, 11 variables meteorológicas y el historial de caudal.

El punto de partida es el método ASDA de Tian et al., propuesto para pronóstico urbano de lluvia y escorrentía con datos cada 10 minutos y horizontes de hasta seis pasos. El artículo combina descomposición adaptativa de la serie, mecanismos de atención y redes recurrentes. En este proyecto adaptamos esas ideas a datos horarios y a un horizonte de 48 horas; por ello, los resultados no son una reproducción directa de los reportados en el paper.

**Referencia:** Tian et al., [Urban real-time rainfall-runoff prediction using adaptive SSA-decomposition with dual attention](https://doi.org/10.1016/j.jhydrol.2025.132701), *Journal of Hydrology* (2025).

## Enfoque

Se compararon tres modelos, usando el mismo conjunto de datos:

- **E1 — LSTM base:** referencia sin descomposición ni atención.
- **E2 — ASDA adaptado:** Adaptive SSA, atención BQA, dos ramas LSTM, TSA y RSD Loss.
- **E3 — Ablación:** variante sin los módulos de atención BQA y TSA.

La SSA se aplicó únicamente al caudal histórico de cada muestra, sin usar las 48 horas futuras. Se calcularon las estadísticas de normalización con train, se seleccionó el modelo por RMSE de validation y se evaluó una sola vez sobre test.

## Resultados

El conjunto completo incluyó 254,000 muestras de train, 18,142 de validation y 27,983 de test. En validation, E1 obtuvo el menor RMSE:

| Modelo | RMSE validation (mm/h) | MAE validation (mm/h) |
|---|---:|---:|
| E1 — LSTM base | **0.10373** | **0.02548** |
| E2 — ASDA adaptado | 0.13549 | 0.04234 |
| E3 — Sin atención | 0.11498 | 0.02859 |

Por ese criterio se seleccionó E1. En test obtuvo **RMSE 0.10226**, **MAE 0.02699**, **NSE 0.62624** y **KGE 0.70264**. El desempeño agregado fue similar al de validation, pero no reflejó bien los extremos: para la muestra con el mayor pico, el valor real alcanzó **16.27951 mm/h** y el máximo pronosticado fue **0.23015 mm/h**.

El PBIAS calculado fue **1651.69 %** en test. En esta implementación es una media de errores porcentuales por observación, sensible a caudales reales cercanos a cero; debe interpretarse con cautela y no compararse directamente con variantes agregadas de la métrica.

## Conclusión

En este conjunto de datos, la adaptación ASDA no superó a la LSTM base en RMSE de validation. Los resultados muestran que incorporar componentes más complejos no garantiza una mejora y que las métricas globales pueden ocultar errores importantes en eventos extremos. La ejecución registrada duró **4.26 horas**, reutilizando los precálculos disponibles.

## Reproducción

El análisis, la configuración y las visualizaciones están en [`notebooks/Lab2.ipynb`](notebooks/Lab2.ipynb). Para reproducirlo se requieren los archivos de datos proporcionados para el laboratorio (`metadata.json`, `train.h5`, `test.h5` y `test_targets.csv`) en `data/`, además de Python 3.11, PyTorch, NumPy, pandas, matplotlib y h5py. Los datos y artefactos generados no se incluyen en el repositorio.
