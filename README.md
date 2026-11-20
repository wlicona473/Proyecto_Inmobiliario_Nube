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

Tablero interactivo en Dash que responde la pregunta de negocio: ¿cómo estimar el valor monetario de las transacciones inmobiliarias de Bogotá D.C. y Cundinamarca (2015–2023) a partir de sus características, dado que el 72,9 % de los registros no tiene valor?

Tiene dos pestañas:
- **Estimar valor y factores**: formulario con las 8 variables del modelo, valor estimado (pesos constantes de dic-2018), rango orientativo, factores que explican cada predicción e importancia global.
- **Dashboard exploratorio**: filtros, indicadores, gráficas y tabla de matrículas priorizadas.

El tablero consulta la API del modelo (`api/`, FastAPI, puerto 8001) y se publica en el puerto 8050.

### Ejecutar con Docker (recomendado)

```bash
docker compose up --build
```

Abrir http://localhost:8050 (tablero) y http://localhost:8001/health (API).

Cada servicio también se puede construir por separado:

```bash
docker build -t re-monitor-api ./api
docker run -p 8001:8001 -e PORT=8001 re-monitor-api

docker build -t re-monitor-dash ./tablero
docker run -p 8050:8050 -e PORT=8050 -e API_URL=IP_DE_LA_API -e API_PORT=8001 re-monitor-dash
```

### Ejecutar sin Docker

```bash
# Terminal 1: API
cd api
pip install -r requirements.txt
uvicorn main:app --port 8001

# Terminal 2: tablero
cd tablero
pip install -r requirements.txt
API_URL=localhost API_PORT=8001 python app.py
```

En PowerShell: `$env:API_URL="localhost"; $env:API_PORT="8001"`.

### Variables de entorno

| Variable | Servicio | Uso |
|---|---|---|
| `PORT` | API y tablero | Puerto del servicio (8001 y 8050) |
| `API_URL` | Tablero | Dirección de la API (IP o nombre) |
| `API_PORT` | Tablero | Puerto de la API |

### Archivos de datos del tablero

`cubo.csv`, `kpis.csv` y `top_riesgo.csv` son resúmenes del parquet procesado. `catalogos.json` trae los nombres de municipios y actos jurídicos, y `metricas_modelo.json` las métricas que muestra la tarjeta «Calidad del modelo».

Limitaciones: el modelo reconoce 38 de 117 municipios, el valor es una estimación orientativa (no un avalúo) y los factores explican el modelo, no demuestran causalidad.
