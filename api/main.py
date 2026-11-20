"""
API de RE-MONITOR: sirve el modelo XGBoost (trial 24) para estimar el valor monetario
de una transacción inmobiliaria de Bogotá D.C. y Cundinamarca.

Importante: el modelo predice LN(VALOR_REAL), donde VALOR_REAL es el valor registrado
ajustado por IPC a pesos constantes de diciembre de 2018. Por eso la API devuelve el
logaritmo y su equivalente en pesos constantes de 2018.

Endpoints:
  GET  /health       estado del servicio
  GET  /opciones     categorías que el modelo conoce (municipios, zonas, actos)
  GET  /importancia  importancia global de cada variable (% de la ganancia)
  POST /predict      estimación + aporte de cada variable a la predicción
Documentación interactiva: /docs
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "modelo" / "modelo_xgboost.pkl"))

VARS = [
    "YEAR_RADICA", "DIVIPOLA", "TIPO_PREDIO_ZONA", "COD_NATUJUR",
    "REGISTROS_PREVIOS", "DIAS_DESDE_ULTIMO_REGISTRO",
    "ANIOS_DESDE_PRIMER_REGISTRO", "ANIOS_CON_ACTIVIDAD_PREVIA",
]
estado = {}


def _variable_de(nombre_columna: str):
    """'categoricas__DIVIPOLA_11001' -> 'DIVIPOLA' (agrupa las columnas one-hot)."""
    limpio = nombre_columna.split("__", 1)[1]
    return next((v for v in VARS if limpio.startswith(v)), None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipe = joblib.load(MODEL_PATH)
    pre, reg = pipe.named_steps["preprocesamiento"], pipe.named_steps["modelo"]
    ohe = pre.named_transformers_["categoricas"].named_steps["codificacion"]
    estado.update(
        pre=pre,
        reg=reg,
        mapa=[_variable_de(n) for n in pre.get_feature_names_out()],
        opciones={v: [str(c) for c in cats]
                  for v, cats in zip(["DIVIPOLA", "TIPO_PREDIO_ZONA", "COD_NATUJUR"], ohe.categories_)},
    )
    yield


app = FastAPI(title="RE-MONITOR API", version="1.0", lifespan=lifespan)


class Entrada(BaseModel):
    YEAR_RADICA: int = Field(..., ge=2015, le=2023, description="Año de radicación")
    DIVIPOLA: str = Field(..., description="Código DIVIPOLA del municipio, p. ej. '11001'")
    TIPO_PREDIO_ZONA: str = Field(..., description="RURAL, URBANO o SIN INFORMACION")
    COD_NATUJUR: str = Field(..., description="Código de naturaleza jurídica del acto, p. ej. '125'")
    REGISTROS_PREVIOS: float = Field(..., ge=0)
    DIAS_DESDE_ULTIMO_REGISTRO: float = Field(..., ge=0)
    ANIOS_DESDE_PRIMER_REGISTRO: float = Field(..., ge=0)
    ANIOS_CON_ACTIVIDAD_PREVIA: float = Field(..., ge=0)


@app.get("/health")
def health():
    return {"status": "ok", "modelo": MODEL_PATH.name}


@app.get("/opciones")
def opciones():
    return estado["opciones"]


@app.get("/importancia")
def importancia():
    imp = dict.fromkeys(VARS, 0.0)
    for var, g in zip(estado["mapa"], estado["reg"].feature_importances_):
        if var:
            imp[var] += float(g)
    total = sum(imp.values()) or 1.0
    filas = [{"variable": v, "importancia_pct": 100 * g / total} for v, g in imp.items()]
    return sorted(filas, key=lambda f: -f["importancia_pct"])


@app.post("/predict")
def predict(entrada: Entrada):
    datos = entrada.model_dump()
    X = pd.DataFrame([datos], columns=VARS)
    Xt = estado["pre"].transform(X)

    # contribuciones por columna (última = valor base); agrupadas por variable original
    contrib = estado["reg"].get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[0]
    base, contrib = float(contrib[-1]), contrib[:-1]
    agg = dict.fromkeys(VARS, 0.0)
    for var, c in zip(estado["mapa"], contrib):
        if var:
            agg[var] += float(c)

    log_valor = base + sum(agg.values())
    avisos = []
    for var in ("DIVIPOLA", "COD_NATUJUR", "TIPO_PREDIO_ZONA"):
        if datos[var] not in estado["opciones"][var]:
            avisos.append(f"{var}='{datos[var]}' no fue visto en el entrenamiento; "
                          "la estimación no incorpora su efecto.")
    return {
        "log_valor": log_valor,
        "valor_pesos_2018": float(np.expm1(log_valor)),
        "base_log": base,
        "contribuciones": [{"variable": v, "aporte_log": a} for v, a in agg.items()],
        "avisos": avisos,
    }
