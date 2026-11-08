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

# Verificación
if not RUTA_DATOS.exists():
    raise FileNotFoundError(f"No se encontró el dataset: {RUTA_DATOS}")

if not RUTA_IPC.exists():
    raise FileNotFoundError(f"No se encontró el IPC: {RUTA_IPC}")

print("Directorio del proyecto:", PROYECTO)
print("Dataset disponible:", RUTA_DATOS.exists())
print("IPC disponible:", RUTA_IPC.exists())


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

print("\nDimensiones del dataset:", df.shape)
print("\nColumnas:")
print(df.columns.tolist())


# %% Inspección básica

print("\nTipos de datos:")
print(df.dtypes)

print("\nValores faltantes:")
print(df.isna().sum())

print("\nPrimeros registros:")
print(df.head())



# %% Conversión y revisión de VALOR

# Limpiar separadores de miles y espacios
valor_limpio = (
    df["VALOR"]
    .str.strip()
    .str.replace(",", "", regex=False)
)

# Convertir a formato numérico
df["VALOR_NUM"] = pd.to_numeric(
    valor_limpio,
    errors="coerce"
)

# Identificar valores que no pudieron convertirse
no_convertidos = (
    df["VALOR"].notna() &
    df["VALOR_NUM"].isna()
)

print("Valores originales no nulos:", df["VALOR"].notna().sum())
print("Valores convertidos:", df["VALOR_NUM"].notna().sum())
print("Valores no convertidos:", no_convertidos.sum())


df["VALOR_NUM"].dtype

# Convertir Float64 nullable a float64
df["VALOR_NUM"] = df["VALOR_NUM"].astype("float64")

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
fechas = df["FECHA_RADICA_TEXTO"].str.strip()

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

ipc.dtypes

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

print(df["VALOR_REAL"].describe())

# %%  Los datos para la regresión

df_modelo = df.loc[
    df["VALOR_REAL"].notna()
].copy()

print("Dimensiones de la muestra:", df_modelo.shape)

print("\nValores faltantes en VALOR_REAL:")
print(df_modelo["VALOR_REAL"].isna().sum())

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

# %% Correspondencia entre COD_NATUJUR y NOMBRE_NATUJUR

# Número de nombres asociados a cada código
nombres_por_codigo = (
    df_modelo.groupby("COD_NATUJUR", observed=True)["NOMBRE_NATUJUR"]
    .nunique()
)

# Número de códigos asociados a cada nombre
codigos_por_nombre = (
    df_modelo.groupby("NOMBRE_NATUJUR", observed=True)["COD_NATUJUR"]
    .nunique()
)

print("Códigos asociados a más de un nombre:")
print(nombres_por_codigo[nombres_por_codigo > 1])

print("\nNombres asociados a más de un código:")
print(codigos_por_nombre[codigos_por_nombre > 1])

correspondencia = (
    df_modelo.groupby(
        ["COD_NATUJUR", "NOMBRE_NATUJUR"],
        observed=True
    )
    .size()
    .reset_index(name="registros")
)

nombres_repetidos = codigos_por_nombre[
    codigos_por_nombre > 1
].index

detalle_repetidos = correspondencia[
    correspondencia["NOMBRE_NATUJUR"].isin(nombres_repetidos)
].sort_values(["NOMBRE_NATUJUR", "COD_NATUJUR"])

print(detalle_repetidos.to_string(index=False))


# %% Frecuencia de códigos de naturaleza jurídica

frecuencia_codigos = (
    df_modelo["COD_NATUJUR"]
    .value_counts()
)


# Eliminar categorías sin observaciones
frecuencia_codigos = frecuencia_codigos[
    frecuencia_codigos > 0
]

print("Códigos realmente observados:", len(frecuencia_codigos))

print("\nDistribución de frecuencias:")
print(
    frecuencia_codigos.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\n20 códigos menos frecuentes:")
print(frecuencia_codigos.tail(20))

# %% Concentración de registros por código

total_registros = frecuencia_codigos.sum()

for umbral in [1, 5, 10, 20, 50, 100]:
    cantidad = (frecuencia_codigos < umbral).sum()
    registros = frecuencia_codigos[
        frecuencia_codigos < umbral
    ].sum()

    print(
        f"Códigos con menos de {umbral} registros: "
        f"{cantidad} ({cantidad / len(frecuencia_codigos):.2%})"
    )

    print(
        f"Registros correspondientes: "
        f"{registros:,} ({registros / total_registros:.4%})\n"
    )


# %% Concentración de códigos de naturaleza jurídica

frecuencia_codigos = df_modelo["COD_NATUJUR"].value_counts()
frecuencia_codigos = frecuencia_codigos[frecuencia_codigos > 0]

frecuencia_ordenada = frecuencia_codigos.sort_values(
    ascending=False
)

acumulado = frecuencia_ordenada.cumsum() / frecuencia_ordenada.sum()

print("\nConcentración de códigos jurídicos:")
for n in [1, 3, 5, 10, 20, 30]:
    print(
        f"Top {n}: {acumulado.iloc[n - 1]:.2%}"
    )

# %% Historial de transacciones por matrícula

transacciones_matricula = (
    df_modelo.groupby("MATRICULA", observed=True)
    .size()
)

print("Matrículas únicas:", transacciones_matricula.size)

print("\nEstadísticas de transacciones por matrícula:")
print(transacciones_matricula.describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
))

print("\nDistribución de transacciones por matrícula:")
print(
    transacciones_matricula.value_counts()
    .sort_index()
    .head(15)
)

print(
    "\nMatrículas con una sola transacción:",
    (transacciones_matricula == 1).sum()
)

print(
    "Matrículas con más de una transacción:",
    (transacciones_matricula > 1).sum()
)


# %% Revisión de NUM_ANOTACION

print("Tipo de dato:", df_modelo["NUM_ANOTACION"].dtype)
print("Registros:", len(df_modelo))
print("Valores faltantes:", df_modelo["NUM_ANOTACION"].isna().sum())
print("Valores únicos:", df_modelo["NUM_ANOTACION"].nunique())

# Limpiar espacios sin modificar la columna original
anotacion = df_modelo["NUM_ANOTACION"].astype("string").str.strip()

# Revisar formato
es_numerico = anotacion.str.fullmatch(r"\d+", na=False)

print("\nValores exclusivamente numéricos:")
print(es_numerico.sum())

print("\nValores con caracteres diferentes a números:")
print((~es_numerico & anotacion.notna()).sum())

print("\nLongitud de los valores:")
print(anotacion.str.len().describe())

# Intentar conversión numérica 
anotacion_num = pd.to_numeric(anotacion, errors="coerce")

print("\nResumen de conversión numérica:")
print("Convertidos:", anotacion_num.notna().sum())
print("No convertidos:", anotacion_num.isna().sum())

if anotacion_num.notna().any():
    print("\nEstadísticas numéricas:")
    print(anotacion_num.describe())

# Frecuencias más comunes
print("\n20 valores más frecuentes:")
print(anotacion.value_counts().head(20))

# Ejemplos de valores distintos
print("\n30 valores diferentes de NUM_ANOTACION:")
print(anotacion.dropna().drop_duplicates().head(30).to_list())

# %% Valores no numéricos de NUM_ANOTACION

valores_no_numericos = anotacion[~es_numerico].value_counts()

print("Cantidad de valores diferentes:")
print(len(valores_no_numericos))

print("\nFrecuencias:")
print(valores_no_numericos.head(30))

anotacion_limpia = (
    df_modelo["NUM_ANOTACION"]
    .astype("string")
    .str.strip()
    .str.replace(",", "", regex=False)
)



# %% Secuencia de anotaciones por matrícula

revision_anotaciones = df_modelo[
    ["MATRICULA", "FECHA_RADICA", "NUM_ANOTACION"]
].copy()

revision_anotaciones["ANOTACION_NUM"] = pd.to_numeric(
    revision_anotaciones["NUM_ANOTACION"],
    errors="coerce"
)

# Identificar matrículas con tres o más transacciones
frecuencia_matriculas = (
    revision_anotaciones.groupby("MATRICULA", observed=True)
    .size()
)

matriculas_recurrentes = frecuencia_matriculas[
    frecuencia_matriculas >= 3
]

# Seleccionar cinco matrículas para inspección
muestra_matriculas = matriculas_recurrentes.sample(
    n=min(5, len(matriculas_recurrentes)),
    random_state=42
)

for matricula in muestra_matriculas.index:
    print(f"\nMATRÍCULA: {matricula}")

    detalle = (
        revision_anotaciones[
            revision_anotaciones["MATRICULA"] == matricula
        ]
        .sort_values("FECHA_RADICA")
    )

    print(
        detalle.to_string(index=False)
    )

revision_anotaciones["ANOTACION_NUM"] = pd.to_numeric(
    anotacion_limpia,
    errors="coerce"
)

print("Total de registros:", len(revision_anotaciones))
print("Convertidos:", revision_anotaciones["ANOTACION_NUM"].notna().sum())
print("No convertidos:", revision_anotaciones["ANOTACION_NUM"].isna().sum())

print("\nValores que no pudieron convertirse:")
print(
    df_modelo.loc[
        revision_anotaciones["ANOTACION_NUM"].isna(),
        "NUM_ANOTACION"
    ].value_counts().head(20)
)

# %% Validación cronológica de NUM_ANOTACION

orden = revision_anotaciones.sort_values(
    ["MATRICULA", "FECHA_RADICA", "ANOTACION_NUM"]
).copy()

orden["ANOTACION_ANTERIOR"] = (
    orden.groupby("MATRICULA", observed=True)["ANOTACION_NUM"]
    .shift(1)
)

orden["FECHA_ANTERIOR"] = (
    orden.groupby("MATRICULA", observed=True)["FECHA_RADICA"]
    .shift(1)
)

# Comparar solamente registros de fechas diferentes
comparables = orden[
    orden["FECHA_RADICA"] > orden["FECHA_ANTERIOR"]
].copy()

# Identificar secuencias que retroceden o no aumentan
retrocesos = comparables[
    comparables["ANOTACION_NUM"] <
    comparables["ANOTACION_ANTERIOR"]
]

repeticiones = comparables[
    comparables["ANOTACION_NUM"] ==
    comparables["ANOTACION_ANTERIOR"]
]

print("Comparaciones entre fechas diferentes:", len(comparables))
print("Retrocesos en la numeración:", len(retrocesos))
print("Repeticiones entre fechas diferentes:", len(repeticiones))

print("\nEjemplos de retrocesos:")
print(
    retrocesos[
        ["MATRICULA", "FECHA_ANTERIOR", "ANOTACION_ANTERIOR",
         "FECHA_RADICA", "ANOTACION_NUM"]
    ].head(20).to_string(index=False)
)

# %% Matrículas con inconsistencias en NUM_ANOTACION

matriculas_retrocesos = retrocesos["MATRICULA"].nunique()
matriculas_repeticiones = repeticiones["MATRICULA"].nunique()

total_matriculas = df_modelo["MATRICULA"].nunique()

print("Total de matrículas:", total_matriculas)

print(
    "Matrículas con retrocesos:",
    matriculas_retrocesos,
    f"({matriculas_retrocesos / total_matriculas:.2%})"
)

print(
    "Matrículas con repeticiones:",
    matriculas_repeticiones,
    f"({matriculas_repeticiones / total_matriculas:.2%})"
)

# Revisar ejemplos de números repetidos en fechas diferentes
print("\nEjemplos de repeticiones:")
print(
    repeticiones[
        [
            "MATRICULA",
            "FECHA_ANTERIOR",
            "ANOTACION_ANTERIOR",
            "FECHA_RADICA",
            "ANOTACION_NUM"
        ]
    ].head(15).to_string(index=False)
)


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

print("Registros de modelamiento:", len(df_modelo))
print("Registros históricos:", len(hist_base))
print("Combinaciones matrícula-fecha:", len(hist_diario))

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

print("\nValores faltantes:")
print(
    df_modelo_hist[
        [
            "REGISTROS_PREVIOS",
            "DIAS_DESDE_ULTIMO_REGISTRO",
            "ANIOS_DESDE_PRIMER_REGISTRO",
            "ANIOS_CON_ACTIVIDAD_PREVIA"
        ]
    ].isna().sum()
)

# %% Distribución de registros por matrícula

registros_por_matricula = (
    df_modelo.groupby("MATRICULA", observed=True)
    .size()
)

print("Total de matrículas:", len(registros_por_matricula))

print("\nDistribución de registros por matrícula:")
print(registros_por_matricula.value_counts().sort_index().head(20))

print("\nEstadísticas descriptivas:")
print(registros_por_matricula.describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
))

# %% Concentración de matrículas según cantidad de registros

total_matriculas = len(registros_por_matricula)

for limite in [1, 2, 3, 5, 10]:
    cantidad = (registros_por_matricula <= limite).sum()

    print(
        f"Matrículas con hasta {limite} registro(s): "
        f"{cantidad:,} "
        f"({cantidad / total_matriculas:.2%})"
    )

# %% Participación de las transacciones según historial de matrícula

resumen_matriculas = pd.DataFrame({
    "registros_por_matricula": registros_por_matricula
})

resumen_matriculas["grupo"] = np.select(
    [
        resumen_matriculas["registros_por_matricula"] == 1,
        resumen_matriculas["registros_por_matricula"] == 2,
        resumen_matriculas["registros_por_matricula"] == 3,
        resumen_matriculas["registros_por_matricula"] >= 4
    ],
    [
        "1 registro",
        "2 registros",
        "3 registros",
        "4 o más registros"
    ],
    default="Sin clasificar"
)

resumen_grupos = (
    resumen_matriculas.groupby("grupo", observed=True)
    .agg(
        matriculas=("registros_por_matricula", "size"),
        transacciones=("registros_por_matricula", "sum")
    )
)

resumen_grupos["porcentaje_matriculas"] = (
    resumen_grupos["matriculas"] / len(registros_por_matricula) * 100
)

resumen_grupos["porcentaje_transacciones"] = (
    resumen_grupos["transacciones"] / len(df_modelo) * 100
)

print(resumen_grupos)    
    
# %% Estadísticas descriptivas de variables históricas

columnas_hist = [
    "REGISTROS_PREVIOS",
    "DIAS_DESDE_ULTIMO_REGISTRO",
    "ANIOS_DESDE_PRIMER_REGISTRO",
    "ANIOS_CON_ACTIVIDAD_PREVIA"
]

print(df_modelo_hist[columnas_hist].describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
))

print("\nValores faltantes:")
print(df_modelo_hist[columnas_hist].isna().sum())

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

TARGET = "VALOR_REAL"

X = df_modelo_hist[FEATURES].copy()
y = df_modelo_hist[TARGET].copy()

print("Dimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

print("\nTipos de variables:")
print(X.dtypes)

print("\nValores faltantes:")
print(X.isna().sum())


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

from sklearn.tree import DecisionTreeRegressor

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

preprocesamiento = ColumnTransformer(
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
                ("imputacion", SimpleImputer(strategy="most_frequent")),
                ("codificacion", OneHotEncoder(handle_unknown="ignore"))
            ]),
            VARIABLES_CATEGORICAS
        )
    ],
    remainder="drop"
)

# Ajustar exclusivamente con entrenamiento
X_train_proc = preprocesamiento.fit_transform(X_train)

# Transformar validación y prueba
X_valid_proc = preprocesamiento.transform(X_valid)
X_test_proc = preprocesamiento.transform(X_test)

print("Entrenamiento transformado:", X_train_proc.shape)
print("Validación transformada:", X_valid_proc.shape)
print("Prueba transformada:", X_test_proc.shape)

print("\nTipo de matriz:", type(X_train_proc))


# %% Búsqueda de hiperparámetros del árbol

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

parametros = {
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [100, 500],
    "min_samples_leaf": [50, 200, 500],
    "criterion": ["squared_error"]
}

combinaciones = list(ParameterGrid(parametros))

resultados_arbol = []

for i, params in enumerate(combinaciones, start=1):

    arbol = DecisionTreeRegressor(
        **params,
        random_state=42
    )

    arbol.fit(X_train_proc, y_train)

    pred_valid = arbol.predict(X_valid_proc)

    mae = mean_absolute_error(y_valid, pred_valid)

    rmse = np.sqrt(
        mean_squared_error(y_valid, pred_valid)
    )

    r2 = r2_score(y_valid, pred_valid)

    resultados_arbol.append({
        "configuracion": i,
        "max_depth": params["max_depth"],
        "min_samples_split": params["min_samples_split"],
        "min_samples_leaf": params["min_samples_leaf"],
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(
        f"Configuración {i}/{len(combinaciones)} | "
        f"MAE: {mae:,.2f} | "
        f"RMSE: {rmse:,.2f} | "
        f"R2: {r2:.4f}"
    )

resultados_arbol = pd.DataFrame(resultados_arbol)

resultados_arbol = resultados_arbol.sort_values(
    "MAE"
).reset_index(drop=True)

print("\nMejores configuraciones:")
print(resultados_arbol.head(10).to_string(index=False))

# %% Modelos de referencia (Baseline)

from sklearn.dummy import DummyRegressor

modelos_base = {
    "Media": DummyRegressor(strategy="mean"),
    "Mediana": DummyRegressor(strategy="median")
}

resultados_base = []

for nombre, modelo in modelos_base.items():

    modelo.fit(X_train_proc, y_train)

    pred = modelo.predict(X_valid_proc)

    resultados_base.append({
        "Modelo": nombre,
        "MAE": mean_absolute_error(y_valid, pred),
        "RMSE": np.sqrt(mean_squared_error(y_valid, pred)),
        "R2": r2_score(y_valid, pred)
    })

resultados_base = pd.DataFrame(resultados_base)

print(resultados_base.to_string(index=False))

# %% Evaluación del mejor árbol en entrenamiento y validación

mejor = resultados_arbol.iloc[0]

arbol_base = DecisionTreeRegressor(
    max_depth=int(mejor["max_depth"]),
    min_samples_split=int(mejor["min_samples_split"]),
    min_samples_leaf=int(mejor["min_samples_leaf"]),
    criterion="squared_error",
    random_state=42
)

arbol_base.fit(X_train_proc, y_train)

pred_train = arbol_base.predict(X_train_proc)
pred_valid = arbol_base.predict(X_valid_proc)

comparacion_arbol = pd.DataFrame({
    "Conjunto": ["Entrenamiento", "Validación"],
    "MAE": [
        mean_absolute_error(y_train, pred_train),
        mean_absolute_error(y_valid, pred_valid)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_train, pred_train)),
        np.sqrt(mean_squared_error(y_valid, pred_valid))
    ],
    "R2": [
        r2_score(y_train, pred_train),
        r2_score(y_valid, pred_valid)
    ]
})

print(comparacion_arbol.to_string(index=False))

print("\nProfundidad real:", arbol_base.get_depth())
print("Número de hojas:", arbol_base.get_n_leaves())


# %% Distribución del objetivo por conjunto

for nombre, objetivo in [
    ("Entrenamiento", y_train),
    ("Validación", y_valid),
    ("Prueba", y_test)
]:

    print(f"\n{nombre}")
    print(
        objetivo.describe(
            percentiles=[0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
        )
    )
    
    
# %% Distribución de VALOR_REAL por año

resumen_anual_valor = (
    df_modelo_hist.groupby("YEAR_RADICA", observed=True)["VALOR_REAL"]
    .agg(
        registros="count",
        media="mean",
        mediana="median",
        minimo="min",
        maximo="max",
        p90=lambda x: x.quantile(0.90),
        p95=lambda x: x.quantile(0.95),
        p99=lambda x: x.quantile(0.99)
    )
)

print(resumen_anual_valor)

# %% Evolución anual del valor inmobiliario real

resumen_anual_valor["mediana"].plot(
    figsize=(10, 5),
    marker="o"
)

plt.title("Evolución anual de la mediana del valor inmobiliario real")
plt.xlabel("Año de radicación")
plt.ylabel("Valor real (pesos de diciembre de 2018)")
plt.grid(alpha=0.3)
plt.ticklabel_format(axis="y", style="plain")
plt.tight_layout()
plt.show()

# %% Detección de valores extremos por año

auditoria_extremos = df_modelo_hist[
    ["YEAR_RADICA", "VALOR_REAL"]
].copy()

# Transformación logarítmica para reducir la asimetría
auditoria_extremos["LOG_VALOR_REAL"] = np.log1p(
    auditoria_extremos["VALOR_REAL"]
)

# Cuartiles del logaritmo por año
limites_anuales = (
    auditoria_extremos
    .groupby("YEAR_RADICA", observed=True)["LOG_VALOR_REAL"]
    .agg(
        Q1=lambda x: x.quantile(0.25),
        Q3=lambda x: x.quantile(0.75)
    )
    .reset_index()
)

# Rango intercuartílico y límites
limites_anuales["IQR"] = (
    limites_anuales["Q3"] - limites_anuales["Q1"]
)

limites_anuales["LIMITE_1_5_IQR"] = (
    limites_anuales["Q3"] + 1.5 * limites_anuales["IQR"]
)

limites_anuales["LIMITE_3_IQR"] = (
    limites_anuales["Q3"] + 3 * limites_anuales["IQR"]
)

# Incorporar límites a cada registro
auditoria_extremos = auditoria_extremos.merge(
    limites_anuales[
        ["YEAR_RADICA", "LIMITE_1_5_IQR", "LIMITE_3_IQR"]
    ],
    on="YEAR_RADICA",
    how="left",
    validate="many_to_one"
)

# Marcar posibles extremos
auditoria_extremos["EXTREMO_1_5_IQR"] = (
    auditoria_extremos["LOG_VALOR_REAL"]
    > auditoria_extremos["LIMITE_1_5_IQR"]
)

auditoria_extremos["EXTREMO_3_IQR"] = (
    auditoria_extremos["LOG_VALOR_REAL"]
    > auditoria_extremos["LIMITE_3_IQR"]
)

# Resumen anual
resumen_extremos = (
    auditoria_extremos
    .groupby("YEAR_RADICA", observed=True)
    .agg(
        registros=("VALOR_REAL", "size"),
        mediana=("VALOR_REAL", "median"),
        maximo=("VALOR_REAL", "max"),
        extremos_1_5_IQR=("EXTREMO_1_5_IQR", "sum"),
        extremos_3_IQR=("EXTREMO_3_IQR", "sum")
    )
)

resumen_extremos["porcentaje_1_5_IQR"] = (
    resumen_extremos["extremos_1_5_IQR"]
    / resumen_extremos["registros"] * 100
)

resumen_extremos["porcentaje_3_IQR"] = (
    resumen_extremos["extremos_3_IQR"]
    / resumen_extremos["registros"] * 100
)

print(resumen_extremos.to_string())


# %% Dataset sin extremos de 3 IQR

mask_extremos_3 = auditoria_extremos["EXTREMO_3_IQR"].to_numpy()

df_modelo_hist_limpio = df_modelo_hist.loc[
    ~mask_extremos_3
].copy()

print("Dataset original:", df_modelo_hist.shape)
print("Dataset limpio:", df_modelo_hist_limpio.shape)
print("Registros eliminados:", mask_extremos_3.sum())
print(
    "Porcentaje eliminado:",
    mask_extremos_3.mean() * 100
)

# %% División temporal

train_mask_limpio = (
    df_modelo_hist_limpio["YEAR_RADICA"] <= 2020
)

valid_mask_limpio = (
    df_modelo_hist_limpio["YEAR_RADICA"].between(2021, 2022)
)

test_mask_limpio = (
    df_modelo_hist_limpio["YEAR_RADICA"] == 2023
)

X_train_limpio = df_modelo_hist_limpio.loc[
    train_mask_limpio, FEATURES
].copy()

y_train_limpio = df_modelo_hist_limpio.loc[
    train_mask_limpio, TARGET
].copy()

X_valid_limpio = df_modelo_hist_limpio.loc[
    valid_mask_limpio, FEATURES
].copy()

y_valid_limpio = df_modelo_hist_limpio.loc[
    valid_mask_limpio, TARGET
].copy()

X_test_limpio = df_modelo_hist_limpio.loc[
    test_mask_limpio, FEATURES
].copy()

y_test_limpio = df_modelo_hist_limpio.loc[
    test_mask_limpio, TARGET
].copy()

print("Entrenamiento:", X_train_limpio.shape)
print("Validación:", X_valid_limpio.shape)
print("Prueba:", X_test_limpio.shape)

# %% Preprocesamiento

preprocesamiento_limpio = ColumnTransformer(
    transformers=[
        (
            "numericas",
            SimpleImputer(strategy="median", add_indicator=True),
            VARIABLES_NUMERICAS
        ),
        (
            "categoricas",
            Pipeline(steps=[
                ("imputacion", SimpleImputer(strategy="most_frequent")),
                ("codificacion", OneHotEncoder(handle_unknown="ignore"))
            ]),
            VARIABLES_CATEGORICAS
        )
    ],
    remainder="drop"
)

X_train_proc_limpio = preprocesamiento_limpio.fit_transform(
    X_train_limpio
)

X_valid_proc_limpio = preprocesamiento_limpio.transform(
    X_valid_limpio
)

X_test_proc_limpio = preprocesamiento_limpio.transform(
    X_test_limpio
)

print("Entrenamiento:", X_train_proc_limpio.shape)
print("Validación:", X_valid_proc_limpio.shape)
print("Prueba:", X_test_proc_limpio.shape)

# %% Búsqueda de hiperparámetros: árbol de decisión

resultados_arbol_3iqr = []

for i, params in enumerate(combinaciones, start=1):

    arbol = DecisionTreeRegressor(
        **params,
        random_state=42
    )

    arbol.fit(
        X_train_proc_limpio,
        y_train_limpio
    )

    pred_valid = arbol.predict(
        X_valid_proc_limpio
    )

    mae = mean_absolute_error(
        y_valid_limpio,
        pred_valid
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid_limpio,
            pred_valid
        )
    )

    r2 = r2_score(
        y_valid_limpio,
        pred_valid
    )

    resultados_arbol_3iqr.append({
        "combinacion": i,
        **params,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

resultados_arbol_3iqr = pd.DataFrame(
    resultados_arbol_3iqr
)

print(
    resultados_arbol_3iqr
    .sort_values("MAE")
    .to_string(index=False)
)

# %% Modelos con R2 positivo

modelos_r2_positivo = resultados_arbol_3iqr.loc[
    resultados_arbol_3iqr["R2"] > 0
].copy()

print("Modelos con R2 positivo:", len(modelos_r2_positivo))

print(
    modelos_r2_positivo
    .sort_values(["MAE", "RMSE"])
    .to_string(index=False)
)

# %% Entrenamiento del árbol seleccionado

arbol_final = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=5,
    min_samples_leaf=50,
    min_samples_split=100,
    random_state=42
)

arbol_final.fit(
    X_train_proc_limpio,
    y_train_limpio
)

pred_valid_final = arbol_final.predict(
    X_valid_proc_limpio
)

print("MAE:", mean_absolute_error(
    y_valid_limpio, pred_valid_final
))

print("RMSE:", np.sqrt(mean_squared_error(
    y_valid_limpio, pred_valid_final
)))

print("R2:", r2_score(
    y_valid_limpio, pred_valid_final
))

# %% Instalación y llamados para RF
pip install optuna

import optuna

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# %% Función objetivo de Optuna

def objetivo_random_forest(trial):

    parametros = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 150, 400, step=50
        ),

        "max_depth": trial.suggest_categorical(
            "max_depth", [8, 12, 16, 20, None]
        ),

        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 50, log=True
        ),

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 2, 50, log=True
        ),

        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", 0.3, 0.5, 0.7, 1.0]
        ),

        "max_samples": trial.suggest_float(
            "max_samples", 0.6, 1.0
        ),

        "bootstrap": True,
        "criterion": "squared_error",
        "n_jobs": -1,
        "random_state": 42
    }

    modelo = RandomForestRegressor(**parametros)

    modelo.fit(
        X_train_proc_limpio,
        y_train_limpio
    )

    predicciones = modelo.predict(
        X_valid_proc_limpio
    )

    mae = mean_absolute_error(
        y_valid_limpio,
        predicciones
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid_limpio,
            predicciones
        )
    )

    r2 = r2_score(
        y_valid_limpio,
        predicciones
    )

    trial.set_user_attr("MAE", mae)
    trial.set_user_attr("R2", r2)

    return rmse

# %% Optimización Random Forest

estudio_rf = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

estudio_rf.optimize(
    objetivo_random_forest,
    n_trials=30
)

# %% Mejores hiperparámetros


print("\nMejores hiperparámetros:")
print(estudio_rf.best_params)

parametros_rf = {
    "n_estimators": 300,
    "max_depth": 16,
    "min_samples_split": 43,
    "min_samples_leaf": 2,
    "max_features": 0.3,
    "max_samples": 0.6819117474333835,
    "bootstrap": True,
    "criterion": "squared_error",
    "n_jobs": -1,
    "random_state": 42
}

# %% Random Forest seleccionado

rf_final = RandomForestRegressor(
    **parametros_rf
)

rf_final.fit(
    X_train_proc_limpio,
    y_train_limpio
)

pred_valid_rf = rf_final.predict(
    X_valid_proc_limpio
)

metricas_rf = {
    "MAE": mean_absolute_error(
        y_valid_limpio, pred_valid_rf
    ),
    "RMSE": np.sqrt(
        mean_squared_error(
            y_valid_limpio, pred_valid_rf
        )
    ),
    "R2": r2_score(
        y_valid_limpio, pred_valid_rf
    )
}

print("Resultados Random Forest:")
print(metricas_rf)

# %% Importancia de variables

nombres_variables = (
    preprocesamiento_limpio.get_feature_names_out()
)

importancias_rf = pd.DataFrame({
    "variable_codificada": nombres_variables,
    "importancia": rf_final.feature_importances_
})

# Identificar la variable original
importancias_rf["variable"] = (
    importancias_rf["variable_codificada"]
    .str.replace(r"^(numericas|categoricas)__", "", regex=True)
)

importancias_rf["variable"] = (
    importancias_rf["variable"]
    .str.split("_")
    .str[0]
)

# Agrupar las importancias por variable
importancia_variables = (
    importancias_rf.groupby("variable")["importancia"]
    .sum()
    .sort_values(ascending=False)
)

print(importancia_variables)

importancia_variables.sort_values().plot(
    kind="barh",
    figsize=(9, 5),
    title="Importancia de variables - Random Forest"
)

plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# %% Instalación
%pip install xgboost

# %% Librerías

import xgboost as xgb
import optuna

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("Versión XGBoost:", xgb.__version__)

# %% Función objetivo XGBoost

def objetivo_xgboost(trial):

    parametros = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 300, 1000, step=100
        ),

        "max_depth": trial.suggest_int(
            "max_depth", 3, 10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        ),

        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1, 20
        ),

        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        ),

        "gamma": trial.suggest_float(
            "gamma", 0, 5
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-4, 10, log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 20, log=True
        ),

        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 0
    }

    modelo = XGBRegressor(**parametros)

    modelo.fit(
        X_train_proc_limpio,
        y_train_limpio,
        verbose=False
    )

    predicciones = modelo.predict(
        X_valid_proc_limpio
    )

    mae = mean_absolute_error(
        y_valid_limpio,
        predicciones
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid_limpio,
            predicciones
        )
    )

    r2 = r2_score(
        y_valid_limpio,
        predicciones
    )

    trial.set_user_attr("MAE", mae)
    trial.set_user_attr("R2", r2)

    return rmse


# %% Optimización XGBoost

estudio_xgb = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

estudio_xgb.optimize(
    objetivo_xgboost,
    n_trials=30
)

# %% Mejores hiperparámetros XGBoost

print("Mejor RMSE:", estudio_xgb.best_value)

print("\nMejores hiperparámetros:")
print(estudio_xgb.best_params)

# %% XGBoost seleccionado

from xgboost import XGBRegressor

parametros_xgb = estudio_xgb.best_params.copy()

xgb_final = XGBRegressor(
    **parametros_xgb,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

xgb_final.fit(
    X_train_proc_limpio,
    y_train_limpio
)

# Predicciones de validación
pred_valid_xgb = xgb_final.predict(
    X_valid_proc_limpio
)

# Métricas
metricas_xgb = {
    "MAE": mean_absolute_error(
        y_valid_limpio, pred_valid_xgb
    ),
    "RMSE": np.sqrt(
        mean_squared_error(
            y_valid_limpio, pred_valid_xgb
        )
    ),
    "R2": r2_score(
        y_valid_limpio, pred_valid_xgb
    )
}

print("Resultados XGBoost:")
print(metricas_xgb)

# %% Comparación de modelos

comparacion_modelos = pd.DataFrame([
    {
        "Modelo": "Árbol de decisión",
        "MAE": 183793200,  
        "RMSE": 1241958000,
        "R2": 0.009692
    },
    {
        "Modelo": "Random Forest",
        "MAE": 168904588.25920784,
        "RMSE": 1236781825.724068,
        "R2": 0.017929916011236613
    },
    {
        "Modelo": "XGBoost",
        "MAE": metricas_xgb["MAE"],
        "RMSE": metricas_xgb["RMSE"],
        "R2": metricas_xgb["R2"]
    }
])

print(
    comparacion_modelos
    .sort_values("RMSE")
    .to_string(index=False)
)

# %% Importancia de variables XGBoost

nombres_variables = (
    preprocesamiento_limpio.get_feature_names_out()
)

importancias_xgb = pd.DataFrame({
    "variable_codificada": nombres_variables,
    "importancia": xgb_final.feature_importances_
})

print("Número de variables:", len(nombres_variables))
print("Número de importancias:", len(xgb_final.feature_importances_))

print(importancias_xgb.head())

# %% Identificar variables originales

def identificar_variable(nombre):

    if nombre.startswith("numericas__"):
        nombre_limpio = nombre.replace("numericas__", "", 1)

        if nombre_limpio.startswith("missingindicator_"):
            return nombre_limpio.replace(
                "missingindicator_", "", 1
            )

        return nombre_limpio

    elif nombre.startswith("categoricas__"):
        nombre_limpio = nombre.replace("categoricas__", "", 1)

        for variable in VARIABLES_CATEGORICAS:
            if nombre_limpio.startswith(variable + "_"):
                return variable

    return nombre


importancias_xgb["variable"] = (
    importancias_xgb["variable_codificada"]
    .apply(identificar_variable)
)

# Agrupar importancias
importancia_variables_xgb = (
    importancias_xgb.groupby("variable")["importancia"]
    .sum()
    .sort_values(ascending=False)
)

print(importancia_variables_xgb)

# %% Gráfico de importancia de variables XGBoost

importancia_variables_xgb.sort_values().plot(
    kind="barh",
    figsize=(9, 5),
    title="Importancia de variables - XGBoost"
)

plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()