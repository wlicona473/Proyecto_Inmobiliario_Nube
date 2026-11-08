# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 00:15:11 2026

@author: aramimoz
"""
# %% Configuración general
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# Directorio raíz del proyecto
PROYECTO = Path(
    r"C:\Users\arami\Documents\Maestría\Despliegue de soluciones analíticas\Proyecto\Proyecto_Inmobiliario_Nube"
)

# Rutas
RUTA_DATOS = PROYECTO / "transacciones_bogota_cundinamarca.parquet"
RUTA_IPC = PROYECTO / "ipc.xlsx"

REPORTS = PROYECTO / "reports"
FIGURAS = REPORTS / "figures"
MODELOS = PROYECTO / "models"

for carpeta in [REPORTS, FIGURAS, MODELOS]:
    carpeta.mkdir(parents=True, exist_ok=True)

# Verificación de archivos
for ruta in [RUTA_DATOS, RUTA_IPC]:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

print("Directorio del proyecto:", PROYECTO)
print("Archivos de entrada verificados correctamente.")

# %% Lectura de datos

COLS = [
    "MATRICULA",
    "FECHA_RADICA_TEXTO",
    "YEAR_RADICA",
    "DIVIPOLA",
    "DEPARTAMENTO",
    "MUNICIPIO",
    "TIPO_PREDIO_ZONA",
    "NUM_ANOTACION",
    "COD_NATUJUR",
    "NOMBRE_NATUJUR",
    "VALOR"
]

CATS = [
    "MATRICULA",
    "DIVIPOLA",
    "DEPARTAMENTO",
    "MUNICIPIO",
    "TIPO_PREDIO_ZONA",
    "COD_NATUJUR",
    "NOMBRE_NATUJUR"
]

df = pq.read_table(
    RUTA_DATOS,
    columns=COLS,
    read_dictionary=CATS
).to_pandas()

print(f"Dimensiones del dataset: {df.shape}")
print(f"Columnas disponibles: {df.columns.tolist()}")

# %% Conversión y revisión de VALOR

# Limpiar separadores de miles y espacios #!!!revisar si hay distición entre separador de miles y decimal!!!
valor_limpio = (
    df["VALOR"]
    .str.strip()
    .str.replace(",", "", regex=False)
)

# Convertir a numero
df["VALOR_NUM"] = pd.to_numeric(
    valor_limpio,
    errors="coerce"
).astype("float64")

# Identificar valores que no pudieron convertirse
no_convertidos = (
    df["VALOR"].notna() &
    df["VALOR_NUM"].isna()
)

print("Valores originales no nulos:", df["VALOR"].notna().sum())
print("Valores convertidos:", df["VALOR_NUM"].notna().sum())
print("Valores no convertidos:", no_convertidos.sum())

# Clasificar los valores monetarios
df["ESTADO_VALOR"] = np.select(
    [
        df["VALOR_NUM"].isna(),
        df["VALOR_NUM"] == 0,
        df["VALOR_NUM"] > 0,
        df["VALOR_NUM"] < 0
    ],
    [
        "NULO",
        "CERO",
        "POSITIVO",
        "NEGATIVO"
    ],
    default="NO_CLASIFICADO"
)

print("\nDistribución de estados de VALOR:")
print(
    df["ESTADO_VALOR"].value_counts(dropna=False)
)

print("\nPorcentaje por estado:")
print(
    df["ESTADO_VALOR"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

# %% Modo valor real usando IPC

# Normalizar fechas de radicación
fechas = df["FECHA_RADICA_TEXTO"].astype("string").str.strip()

es_iso = fechas.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
es_barras = fechas.str.contains("/", na=False)

df["FECHA_RADICA"] = pd.NaT

df.loc[es_iso, "FECHA_RADICA"] = pd.to_datetime(
    fechas.loc[es_iso],
    format="mixed",
    errors="coerce"
)

df.loc[es_barras, "FECHA_RADICA"] = pd.to_datetime(
    fechas.loc[es_barras],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

# Crear periodo de forma mensual
df["PERIODO"] = df["FECHA_RADICA"].dt.to_period("M")

# Leer IPC
ipc = pd.read_excel(RUTA_IPC)

ipc["PERIODO"] = ipc["Fecha"].dt.to_period("M")

ipc_mensual = ipc[["PERIODO", "IPC"]].copy()

# Integrar IPC mensual
df = df.merge(
    ipc_mensual,
    on="PERIODO",
    how="left",
    validate="many_to_one"
)

print("Dimensiones después de integrar IPC:", df.shape)
print("\nValores faltantes:")
print(df[["FECHA_RADICA", "PERIODO", "IPC"]].isna().sum())

print("\nIPC de diciembre de 2018:")
print(
    df.loc[
        df["PERIODO"] == pd.Period("2018-12"),
        "IPC"
    ].unique()
)

# %% Cálculo del valor real (pesos de diciembre de 2018)

IPC_BASE = 100

df["VALOR_REAL"] = np.nan

condicion = (
    df["VALOR_NUM"].notna() &
    (df["VALOR_NUM"] > 0) &
    df["IPC"].notna()
)

df.loc[condicion, "VALOR_REAL"] = (
    df.loc[condicion, "VALOR_NUM"]
    * IPC_BASE
    / df.loc[condicion, "IPC"]
)

print("Registros con VALOR_REAL:", df["VALOR_REAL"].notna().sum())
print("Registros sin VALOR_REAL:", df["VALOR_REAL"].isna().sum())

# %%  Los datos para la regresión

df_modelo = df.loc[
    df["VALOR_REAL"].notna()
].copy()

print("Dimensiones de la muestra:", df_modelo.shape)

print("\nDistribución temporal:")
print(
    df_modelo["YEAR_RADICA"]
    .value_counts()
    .sort_index()
)

# %% Revisión de variables candidatas

VARIABLES = [
    "YEAR_RADICA",
    "DIVIPOLA",
    "DEPARTAMENTO",
    "MUNICIPIO",
    "TIPO_PREDIO_ZONA",
    "COD_NATUJUR",
    "NOMBRE_NATUJUR",
    "NUM_ANOTACION",
    "MATRICULA"
]

resumen_variables = pd.DataFrame({
    "tipo": df_modelo[VARIABLES].dtypes.astype(str),
    "faltantes": df_modelo[VARIABLES].isna().sum(),
    "categorias_unicas": df_modelo[VARIABLES].nunique()
})

print(resumen_variables)

# %% Base histórica a partir de df_modelo

hist_base = df_modelo.loc[
    df_modelo["FECHA_RADICA"].notna(),
    ["MATRICULA", "FECHA_RADICA"]
].copy()

hist_diario = (
    hist_base.groupby(
        ["MATRICULA", "FECHA_RADICA"],
        observed=True
    )
    .size()
    .reset_index(name="REGISTROS_DIA")
    .sort_values(["MATRICULA", "FECHA_RADICA"])
    .reset_index(drop=True)
)

# %% Variables históricas por matrícula

grupo = hist_diario.groupby("MATRICULA", observed=True)

# Número de transacciones anteriores
hist_diario["REGISTROS_PREVIOS"] = (
    grupo["REGISTROS_DIA"].cumsum()
    - hist_diario["REGISTROS_DIA"]
)

# Fecha del último registro anterior
hist_diario["FECHA_ULTIMO_REGISTRO"] = (
    grupo["FECHA_RADICA"].shift(1)
)

# Días desde la última transacción anterior
hist_diario["DIAS_DESDE_ULTIMO_REGISTRO"] = (
    hist_diario["FECHA_RADICA"]
    - hist_diario["FECHA_ULTIMO_REGISTRO"]
).dt.days

# Años desde el primer registro observado
fecha_primera = grupo["FECHA_RADICA"].transform("min")

hist_diario["ANIOS_DESDE_PRIMER_REGISTRO"] = (
    hist_diario["FECHA_RADICA"] - fecha_primera
).dt.days / 365

# %% Años con actividad previa

hist_base["ANIO"] = hist_base["FECHA_RADICA"].dt.year

anios_matricula = (
    hist_base[["MATRICULA", "ANIO"]]
    .drop_duplicates()
    .sort_values(["MATRICULA", "ANIO"])
    .reset_index(drop=True)
)

anios_matricula["ANIOS_CON_ACTIVIDAD_PREVIA"] = (
    anios_matricula.groupby("MATRICULA", observed=True)
    .cumcount()
)

# %% Integración de variables históricas

columnas_hist = [
    "MATRICULA",
    "FECHA_RADICA",
    "REGISTROS_PREVIOS",
    "DIAS_DESDE_ULTIMO_REGISTRO",
    "ANIOS_DESDE_PRIMER_REGISTRO"
]

df_modelo_hist = df_modelo.merge(
    hist_diario[columnas_hist],
    on=["MATRICULA", "FECHA_RADICA"],
    how="left",
    validate="many_to_one"
)

df_modelo_hist["ANIO"] = (
    df_modelo_hist["FECHA_RADICA"].dt.year
)

df_modelo_hist = df_modelo_hist.merge(
    anios_matricula,
    on=["MATRICULA", "ANIO"],
    how="left",
    validate="many_to_one"
)

print("Dimensiones originales:", df_modelo.shape)
print("Dimensiones con historial:", df_modelo_hist.shape)

VARIABLES_HIST = [
    "REGISTROS_PREVIOS",
    "DIAS_DESDE_ULTIMO_REGISTRO",
    "ANIOS_DESDE_PRIMER_REGISTRO",
    "ANIOS_CON_ACTIVIDAD_PREVIA"
]

print("\nValores faltantes:")
print(df_modelo_hist[VARIABLES_HIST].isna().sum())

# %% Definición de variables predictoras y objetivo

FEATURES = [
    "YEAR_RADICA",
    "DIVIPOLA",
    "TIPO_PREDIO_ZONA",
    "COD_NATUJUR",
    "REGISTROS_PREVIOS",
    "DIAS_DESDE_ULTIMO_REGISTRO",
    "ANIOS_DESDE_PRIMER_REGISTRO",
    "ANIOS_CON_ACTIVIDAD_PREVIA"
]

TARGET = "LOG_VALOR_REAL"

# Variable objetivo en escala logarítmica
df_modelo_hist[TARGET] = np.log(
    df_modelo_hist["VALOR_REAL"]
)

X = df_modelo_hist[FEATURES].copy()
y = df_modelo_hist[TARGET].copy()

print("Dimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

print("\nTipos de variables:")
print(X.dtypes)

print("\nValores faltantes:")
print(X.isna().sum())

print("\nResumen de la variable objetivo:")
print(y.describe())

# %% División temporal de los datos

train_mask = df_modelo_hist["YEAR_RADICA"] <= 2020
valid_mask = df_modelo_hist["YEAR_RADICA"].between(2021, 2022)
test_mask = df_modelo_hist["YEAR_RADICA"] == 2023

X_train = X.loc[train_mask].copy()
y_train = y.loc[train_mask].copy()

X_valid = X.loc[valid_mask].copy()
y_valid = y.loc[valid_mask].copy()

X_test = X.loc[test_mask].copy()
y_test = y.loc[test_mask].copy()

print("Entrenamiento:", X_train.shape)
print("Validación:", X_valid.shape)
print("Prueba:", X_test.shape)

print("\nDistribución temporal:")
print(
    df_modelo_hist.groupby("YEAR_RADICA", observed=True)
    .size()
)

# %% Preprocesamiento

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

VARIABLES_CATEGORICAS = [
    "DIVIPOLA",
    "TIPO_PREDIO_ZONA",
    "COD_NATUJUR"
]

VARIABLES_NUMERICAS = [
    "YEAR_RADICA",
    "REGISTROS_PREVIOS",
    "DIAS_DESDE_ULTIMO_REGISTRO",
    "ANIOS_DESDE_PRIMER_REGISTRO",
    "ANIOS_CON_ACTIVIDAD_PREVIA"
]

def crear_preprocesamiento():

    return ColumnTransformer(
        transformers=[
            (
                "numericas",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                ),
                VARIABLES_NUMERICAS
            ),
            (
                "categoricas",
                Pipeline(steps=[
                    (
                        "imputacion",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "codificacion",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ]),
                VARIABLES_CATEGORICAS
            )
        ],
        remainder="drop"
    )



# %% Función de evaluación

def evaluar_modelo(y_real, y_pred):

    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "R2": r2_score(y_real, y_pred)
    }

# %% Modelos de referencia (Baseline)

modelos_base = {
    "Media": DummyRegressor(strategy="mean"),
    "Mediana": DummyRegressor(strategy="median")
}

resultados_base = []

for nombre, modelo in modelos_base.items():

    modelo.fit(X_train, y_train)

    pred_valid = modelo.predict(X_valid)

    metricas = evaluar_modelo(y_valid, pred_valid)

    resultados_base.append({
        "Modelo": nombre,
        **metricas
    })

resultados_base = (
    pd.DataFrame(resultados_base)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print(resultados_base.to_string(index=False))

# %% Búsqueda de hiperparámetros: árbol de decisión

parametros = {
    "modelo__max_depth": [5, 10, 15, None],
    "modelo__min_samples_split": [100, 500],
    "modelo__min_samples_leaf": [50, 200, 500],
    "modelo__criterion": ["squared_error"]
}

combinaciones = list(ParameterGrid(parametros))

resultados_arbol = []

for i, params in enumerate(combinaciones, start=1):

    pipeline_arbol = Pipeline(steps=[
        ("preprocesamiento", crear_preprocesamiento()),
        ("modelo", DecisionTreeRegressor(random_state=42))
    ])

    pipeline_arbol.set_params(**params)

    pipeline_arbol.fit(X_train, y_train)

    pred_valid = pipeline_arbol.predict(X_valid)

    metricas = evaluar_modelo(y_valid, pred_valid)

    resultados_arbol.append({
        "configuracion": i,
        **params,
        **metricas
    })

    print(
        f"Configuración {i}/{len(combinaciones)} | "
        f"MAE: {metricas['MAE']:.4f} | "
        f"RMSE: {metricas['RMSE']:.4f} | "
        f"R2: {metricas['R2']:.4f}"
    )

resultados_arbol = (
    pd.DataFrame(resultados_arbol)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("\nMejores configuraciones:")
print(resultados_arbol.head(10).to_string(index=False))


# %% Evaluación del mejor árbol

mejor_configuracion = resultados_arbol.iloc[0]

mejores_parametros = {
    "modelo__max_depth": (
        None if pd.isna(mejor_configuracion["modelo__max_depth"])
        else int(mejor_configuracion["modelo__max_depth"])
    ),
    "modelo__min_samples_split": int(
        mejor_configuracion["modelo__min_samples_split"]
    ),
    "modelo__min_samples_leaf": int(
        mejor_configuracion["modelo__min_samples_leaf"]
    ),
    "modelo__criterion": mejor_configuracion["modelo__criterion"]
}

print("Mejores parámetros:")
print(mejores_parametros)

arbol_base = Pipeline(steps=[
    ("preprocesamiento", crear_preprocesamiento()),
    ("modelo", DecisionTreeRegressor(random_state=42))
])

arbol_base.set_params(**mejores_parametros)

arbol_base.fit(X_train, y_train)

# Predicciones
pred_train = arbol_base.predict(X_train)
pred_valid = arbol_base.predict(X_valid)

# Comparación
comparacion_arbol = pd.DataFrame([
    {
        "Conjunto": "Entrenamiento",
        **evaluar_modelo(y_train, pred_train)
    },
    {
        "Conjunto": "Validación",
        **evaluar_modelo(y_valid, pred_valid)
    }
])

print(comparacion_arbol.to_string(index=False))

arbol_entrenado = arbol_base.named_steps["modelo"]

print("\nProfundidad:", arbol_entrenado.get_depth())
print("Número de hojas:", arbol_entrenado.get_n_leaves())

# %% Tratamiento de valores extremos: 3 IQR

# Calcular límites exclusivamente con entrenamiento
Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)

IQR = Q3 - Q1

LIMITE_SUPERIOR = Q3 + 3 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Límite superior:", LIMITE_SUPERIOR)

# Identificar extremos en entrenamiento
mask_extremos_train = y_train > LIMITE_SUPERIOR

# Eliminar extremos únicamente del entrenamiento
X_train_limpio = X_train.loc[
    ~mask_extremos_train
].copy()

y_train_limpio = y_train.loc[
    ~mask_extremos_train
].copy()

# Mantener validación y prueba completas
X_valid_limpio = X_valid.copy()
y_valid_limpio = y_valid.copy()

X_test_limpio = X_test.copy()
y_test_limpio = y_test.copy()

print("\nResultados del tratamiento:")
print("Entrenamiento original:", len(y_train))
print("Entrenamiento limpio:", len(y_train_limpio))
print("Registros eliminados:", mask_extremos_train.sum())

print(
    "Porcentaje eliminado:",
    round(mask_extremos_train.mean() * 100, 2)
)

print("\nValidación conservada:", len(y_valid_limpio))
print("Prueba conservada:", len(y_test_limpio))

# %% Búsqueda de hiperparámetros: árbol con entrenamiento limpio

resultados_arbol_3iqr = []

for i, params in enumerate(combinaciones, start=1):

    pipeline_arbol = Pipeline(steps=[
        ("preprocesamiento", crear_preprocesamiento()),
        ("modelo", DecisionTreeRegressor(random_state=42))
    ])

    pipeline_arbol.set_params(**params)

    pipeline_arbol.fit(
        X_train_limpio,
        y_train_limpio
    )

    pred_valid = pipeline_arbol.predict(
        X_valid_limpio
    )

    metricas = evaluar_modelo(
        y_valid_limpio,
        pred_valid
    )

    resultados_arbol_3iqr.append({
        "configuracion": i,
        **params,
        **metricas
    })

resultados_arbol_3iqr = (
    pd.DataFrame(resultados_arbol_3iqr)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("Mejores configuraciones con entrenamiento limpio:")
print(resultados_arbol_3iqr.head(10).to_string(index=False))

# %% Entrenamiento del árbol seleccionado con entrenamiento limpio

mejor_configuracion_3iqr = resultados_arbol_3iqr.iloc[0]

mejores_parametros_3iqr = {
    "modelo__max_depth": (
        None if pd.isna(mejor_configuracion_3iqr["modelo__max_depth"])
        else int(mejor_configuracion_3iqr["modelo__max_depth"])
    ),
    "modelo__min_samples_split": int(
        mejor_configuracion_3iqr["modelo__min_samples_split"]
    ),
    "modelo__min_samples_leaf": int(
        mejor_configuracion_3iqr["modelo__min_samples_leaf"]
    ),
    "modelo__criterion": mejor_configuracion_3iqr["modelo__criterion"]
}

print(mejores_parametros_3iqr)

arbol_final = Pipeline(steps=[
    ("preprocesamiento", crear_preprocesamiento()),
    ("modelo", DecisionTreeRegressor(random_state=42))
])

arbol_final.set_params(**mejores_parametros_3iqr)

arbol_final.fit(
    X_train_limpio,
    y_train_limpio
)

# Evaluación en entrenamiento y validación
pred_train_final = arbol_final.predict(X_train_limpio)

pred_valid_final = arbol_final.predict(X_valid_limpio)

comparacion_arbol_final = pd.DataFrame([
    {
        "Conjunto": "Entrenamiento",
        **evaluar_modelo(y_train_limpio, pred_train_final)
    },
    {
        "Conjunto": "Validación",
        **evaluar_modelo(y_valid_limpio, pred_valid_final)
    }
])

print(comparacion_arbol_final.to_string(index=False))

# %% Baseline con entrenamiento limpio

baseline_limpio = DummyRegressor(strategy="median")

baseline_limpio.fit(
    X_train_limpio,
    y_train_limpio
)

pred_baseline_limpio = baseline_limpio.predict(
    X_valid_limpio
)

print(
    evaluar_modelo(
        y_valid_limpio,
        pred_baseline_limpio
    )
)


# %% Instalación y llamados para RF

import optuna
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

# %% Función objetivo de Optuna RF

def objetivo_random_forest(trial):

    parametros = {
        "modelo__n_estimators": trial.suggest_int(
            "n_estimators", 150, 400, step=50
        ),

        "modelo__max_depth": trial.suggest_categorical(
            "max_depth", [8, 12, 16, 20, None]
        ),

        "modelo__min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 50, log=True
        ),

        "modelo__min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 2, 50, log=True
        ),

        "modelo__max_features": trial.suggest_categorical(
            "max_features", ["sqrt", 0.3, 0.5, 0.7, 1.0]
        ),

        "modelo__max_samples": trial.suggest_float(
            "max_samples", 0.6, 1.0
        )
    }

    pipeline = Pipeline(steps=[
        ("preprocesamiento", crear_preprocesamiento()),
        ("modelo", RandomForestRegressor(
            bootstrap=True,
            criterion="squared_error",
            n_jobs=-1,
            random_state=42
        ))
    ])

    pipeline.set_params(**parametros)

    pipeline.fit(X_train_limpio, y_train_limpio)

    predicciones = pipeline.predict(X_valid)

    metricas = evaluar_modelo(y_valid, predicciones)

    trial.set_user_attr("MAE", metricas["MAE"])
    trial.set_user_attr("R2", metricas["R2"])

    return metricas["RMSE"]

# %% Optimización Random Forest

estudio_rf = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

estudio_rf.optimize(
    objetivo_random_forest,
    n_trials=30
)

print("Mejor RMSE:", estudio_rf.best_value)
print("\nMejores hiperparámetros:")
print(estudio_rf.best_params)

# %% Random Forest seleccionado

pipeline_rf = Pipeline(steps=[
    ("preprocesamiento", crear_preprocesamiento()),
    ("modelo", RandomForestRegressor(
        bootstrap=True,
        criterion="squared_error",
        n_jobs=-1,
        random_state=42
    ))
])

parametros_rf = {
    f"modelo__{nombre}": valor
    for nombre, valor in estudio_rf.best_params.items()
}

pipeline_rf.set_params(**parametros_rf)

pipeline_rf.fit(X_train_limpio, y_train_limpio)

pred_valid_rf = pipeline_rf.predict(X_valid)

metricas_rf = evaluar_modelo(y_valid, pred_valid_rf)

print("Resultados Random Forest:")
print(metricas_rf)

# %% Función objetivo - XGBoost

def objetivo_xgboost(trial):

    parametros = {
        "modelo__n_estimators": trial.suggest_int(
            "n_estimators", 300, 1000, step=100
        ),

        "modelo__max_depth": trial.suggest_int(
            "max_depth", 3, 10
        ),

        "modelo__learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        ),

        "modelo__min_child_weight": trial.suggest_float(
            "min_child_weight", 1, 20
        ),

        "modelo__subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
        ),

        "modelo__colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        ),

        "modelo__gamma": trial.suggest_float(
            "gamma", 0, 5
        ),

        "modelo__reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-4, 10, log=True
        ),

        "modelo__reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 20, log=True
        )
    }

    pipeline = Pipeline(steps=[
        ("preprocesamiento", crear_preprocesamiento()),
        ("modelo", XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            verbosity=0
        ))
    ])

    pipeline.set_params(**parametros)

    pipeline.fit(X_train_limpio, y_train_limpio)

    predicciones = pipeline.predict(X_valid)

    metricas = evaluar_modelo(y_valid, predicciones)

    trial.set_user_attr("MAE", metricas["MAE"])
    trial.set_user_attr("R2", metricas["R2"])

    return metricas["RMSE"]


# %% Optimización XGBoost

estudio_xgb = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

estudio_xgb.optimize(
    objetivo_xgboost,
    n_trials=30
)

print("Mejor RMSE:", estudio_xgb.best_value)
print("\nMejores hiperparámetros:")
print(estudio_xgb.best_params)


# %% XGBoost final

pipeline_xgb = Pipeline(steps=[
    ("preprocesamiento", crear_preprocesamiento()),
    ("modelo", XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=0
    ))
])

parametros_xgb = {
    f"modelo__{nombre}": valor
    for nombre, valor in estudio_xgb.best_params.items()
}

pipeline_xgb.set_params(**parametros_xgb)

pipeline_xgb.fit(X_train_limpio, y_train_limpio)

pred_valid_xgb = pipeline_xgb.predict(X_valid)

metricas_xgb = evaluar_modelo(y_valid, pred_valid_xgb)

print("Resultados XGBoost:")
print(metricas_xgb)

print("\nVersión XGBoost:", xgb.__version__)


# %% Identificación de variables originales

def identificar_variable(nombre):

    if nombre.startswith("numericas__"):

        nombre_limpio = nombre.replace("numericas__", "", 1)

        if nombre_limpio.startswith("missingindicator_"):
            return nombre_limpio.replace(
                "missingindicator_", "", 1
            )

        return nombre_limpio

    if nombre.startswith("categoricas__"):

        nombre_limpio = nombre.replace("categoricas__", "", 1)

        for variable in VARIABLES_CATEGORICAS:
            if nombre_limpio.startswith(variable + "_"):
                return variable

    return nombre

# %% Importancia de variables - Random Forest

preprocesamiento_rf = pipeline_rf.named_steps["preprocesamiento"]
modelo_rf = pipeline_rf.named_steps["modelo"]

nombres_variables_rf = (
    preprocesamiento_rf.get_feature_names_out()
)

importancias_rf = pd.DataFrame({
    "variable_codificada": nombres_variables_rf,
    "importancia": modelo_rf.feature_importances_
})

importancias_rf["variable"] = (
    importancias_rf["variable_codificada"]
    .apply(identificar_variable)
)

importancia_variables_rf = (
    importancias_rf.groupby("variable")["importancia"]
    .sum()
    .sort_values(ascending=False)
)

print("Importancia de variables - Random Forest:")
print(importancia_variables_rf)

importancia_variables_rf.sort_values().plot(
    kind="barh",
    figsize=(9, 5),
    title="Importancia de variables - Random Forest"
)

plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# %% Importancia de variables - XGBoost

preprocesamiento_xgb = pipeline_xgb.named_steps["preprocesamiento"]
modelo_xgb = pipeline_xgb.named_steps["modelo"]

nombres_variables_xgb = (
    preprocesamiento_xgb.get_feature_names_out()
)

importancias_xgb = pd.DataFrame({
    "variable_codificada": nombres_variables_xgb,
    "importancia": modelo_xgb.feature_importances_
})

importancias_xgb["variable"] = (
    importancias_xgb["variable_codificada"]
    .apply(identificar_variable)
)

importancia_variables_xgb = (
    importancias_xgb.groupby("variable")["importancia"]
    .sum()
    .sort_values(ascending=False)
)

print("Importancia de variables - XGBoost:")
print(importancia_variables_xgb)

importancia_variables_xgb.sort_values().plot(
    kind="barh",
    figsize=(9, 5),
    title="Importancia de variables - XGBoost"
)

plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# %% Comparación de modelos

# Métricas del árbol final sobre validación
metricas_arbol = evaluar_modelo(
    y_valid,
    pred_valid_final
)

comparacion_modelos = pd.DataFrame([
    {
        "Modelo": "Árbol de decisión",
        **metricas_arbol
    },
    {
        "Modelo": "Random Forest",
        **metricas_rf
    },
    {
        "Modelo": "XGBoost",
        **metricas_xgb
    }
])

comparacion_modelos = comparacion_modelos.sort_values(
    "RMSE"
).reset_index(drop=True)

print(comparacion_modelos.to_string(index=False))

