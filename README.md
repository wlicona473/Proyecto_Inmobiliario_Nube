# Proyecto Inmobiliario

Proyecto desarrollado para la asignatura **Despliegue de Soluciones Analíticas**.

## Descripción

El proyecto tiene como propósito el desarrollo y evaluar un prototipo analítico para reconocer y priorizar patrones transaccionales potencialmente anómalos en los registros inmobiliarios en los territorios Bogotá y Cundinamarca, empleando información del Registro de Transacciones Inmobiliarias del IGAC para el periodo 2015–2023.

## Objetivo

Facilitar la identificación de operaciones transaccionales que se aparten del comportamiento esperado de grupos comparables y que puedan requerir una revisión posterior.

## Estructura del repositorio

- `src/`: código fuente del proyecto.
- `notebooks/`: cuadernos de análisis y experimentación.
- `data/`: datos gestionados mediante DVC.

## Versionamiento

- Git y GitHub se utilizan para versionar el código.
- DVC se utiliza para versionar los conjuntos de datos.

## Equipo

- Estudiantes UniAndes MIAD

## Tablero (RE-MONITOR)

Tablero interactivo en Dash, en la carpeta `tablero/`.

Ejecutar:

    cd tablero
    pip install -r requirements.txt
    python app.py

Abrir http://127.0.0.1:8050

Datos: `cubo.csv`, `top_riesgo.csv` y `kpis.csv` son resúmenes del parquet procesado.
`top_riesgo.csv` usa un puntaje de riesgo provisional hasta integrar el modelo del equipo.
`metricas.csv` es opcional: si existe, el tablero muestra las métricas del modelo.