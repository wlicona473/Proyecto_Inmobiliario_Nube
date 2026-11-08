# 
"""
Created on Sun Oct 25 22:07:33 2026

@author: aramimoz
"""
# %% Configuración
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

# Identificar automáticamente la raíz del proyecto
PROYECTO = Path(r"C:\Users\arami\Documents\Maestría\Despliegue de soluciones analíticas\Proyecto\Proyecto_Inmobiliario_Nube")

# Rutas de archivos
RUTA = PROYECTO / "transacciones_bogota_cundinamarca.parquet" 
FIG = PROYECTO / "reports" / "figures"
# Verificar que el dataset exista
if not RUTA.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset en: {RUTA}"
    )


FIG.mkdir(parents=True, exist_ok=True)

print("Directorio del proyecto:", PROYECTO)
print("Dataset encontrado:", RUTA)
def guardar(nombre):
    plt.savefig(FIG / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.show()


# %% Inspección inicial del Parquet

pf = pq.ParquetFile(RUTA)

print("Número de registros:", pf.metadata.num_rows)
print("Número de columnas:", pf.metadata.num_columns)

print("\nEsquema del dataset:")
print(pf.schema_arrow)

# %% Carga liviana
COLS = ["MATRICULA", "FECHA_RADICA_TEXTO", "YEAR_RADICA", "DIVIPOLA", "DEPARTAMENTO", "MUNICIPIO",
        "TIPO_PREDIO_ZONA", "NUM_ANOTACION", "ESTADO_FOLIO", "Dinámica_Inmobiliaria",
        "COD_NATUJUR", "NOMBRE_NATUJUR", "PREDIOS_NUEVOS", "TIENE_VALOR",
        "TIENE_MAS_DE_UN_VALOR", "VALOR"]
CATS = ["MATRICULA", "DIVIPOLA", "DEPARTAMENTO", "MUNICIPIO", "TIPO_PREDIO_ZONA",
        "ESTADO_FOLIO", "COD_NATUJUR", "NOMBRE_NATUJUR"]
df = pq.read_table(RUTA, columns=COLS, read_dictionary=CATS).to_pandas()
print("Dimensiones:", df.shape)
df.info(memory_usage="deep")


# %% Diccionario de datos

descripciones = {
    "MATRICULA": "Identificador de la matrícula inmobiliaria del predio.",
    "FECHA_RADICA_TEXTO": "Fecha de radicación del registro inmobiliario.",
    "YEAR_RADICA": "Año de radicación del registro.",
    "DIVIPOLA": "Código de identificación territorial del municipio.",
    "DEPARTAMENTO": "Departamento o entidad territorial donde se ubica el predio.",
    "MUNICIPIO": "Municipio donde se ubica el predio.",
    "TIPO_PREDIO_ZONA": "Clasificación del tipo de predio o zona.",
    "NUM_ANOTACION": "Número de anotación asociado al folio inmobiliario.",
    "ESTADO_FOLIO": "Estado del folio inmobiliario.",
    "Dinámica_Inmobiliaria": "Indicador relacionado con la dinámica inmobiliaria.",
    "COD_NATUJUR": "Código de la naturaleza jurídica del acto registrado.",
    "NOMBRE_NATUJUR": "Descripción de la naturaleza jurídica del acto.",
    "PREDIOS_NUEVOS": "Indicador o cantidad asociada a la generación de predios nuevos.",
    "TIENE_VALOR": "Indicador de disponibilidad de un valor monetario registrado.",
    "TIENE_MAS_DE_UN_VALOR": "Indicador de presencia de más de un valor monetario.",
    "VALOR": "Valor monetario asociado al registro inmobiliario."
}

usos = {
    "MATRICULA": "Identificador que sirve para construir variables históricas.",
    "FECHA_RADICA_TEXTO": "Variable temporal para ordenar cronológicamente los registros.",
    "YEAR_RADICA": "Variable temporal y criterio para partición de datos.",
    "DIVIPOLA": "predictor territorial.",
    "DEPARTAMENTO": "Caracterización territorial.",
    "MUNICIPIO": "predictor territorial.",
    "TIPO_PREDIO_ZONA": "predictor categórico.",
    "NUM_ANOTACION": "Variable auxiliar para estudiar la secuencia registral.",
    "ESTADO_FOLIO": "Exploración; depende de disponibilidad.",
    "Dinámica_Inmobiliaria": "Caracterización y exploración.",
    "COD_NATUJUR": "Predictor jurídico.",
    "NOMBRE_NATUJUR": "Caracterización e interpretación del acto jurídico.",
    "PREDIOS_NUEVOS": "Exploración; depende de disponibiliada.",
    "TIENE_VALOR": "Análisis de disponibilidad del valor.",
    "TIENE_MAS_DE_UN_VALOR": "Control de calidad y exploración.",
    "VALOR": "Variable objetivo."
}

observaciones = {
    "MATRICULA": "Puede repetirse porque una matrícula puede tener múltiples anotaciones.",
    "FECHA_RADICA_TEXTO": "Se debe convertir a fecha y validar su formato.",
    "YEAR_RADICA": "Contrastar con el año obtenido de la fecha de radicación.",
    "DIVIPOLA": "Verificar consistencia de códigos y correspondencia territorial.",
    "DEPARTAMENTO": "Revisar consistencia de nombres y categorías.",
    "MUNICIPIO": "Revisar consistencia de nombres y categorías.",
    "TIPO_PREDIO_ZONA": "Verificar categorías y valores faltantes.",
    "NUM_ANOTACION": "No asumir que determina por sí solo el orden cronológico.",
    "ESTADO_FOLIO": "Presenta una proporción importante de valores faltantes.",
    "Dinámica_Inmobiliaria": "Revisar definición original del indicador.",
    "COD_NATUJUR": "Variable categórica; verificar correspondencia con el nombre del acto.",
    "NOMBRE_NATUJUR": "Puede presentar alta cardinalidad.",
    "PREDIOS_NUEVOS": "Verificar si representa un indicador o una cantidad.",
    "TIENE_VALOR": "No todos los registros marcados tienen un valor positivo.",
    "TIENE_MAS_DE_UN_VALOR": "Su frecuencia es muy baja en el conjunto analizado.",
    "VALOR": "Está almacenada originalmente como texto; requiere conversión numérica."
}

# Registros del diccionario

registros_diccionario = []

for col in COLS:

    serie = df[col]

    ejemplos = serie.dropna().astype(str).drop_duplicates().head(5).tolist()

    registros_diccionario.append({
        "Variable": col,
        "Descripcion": descripciones.get(col, "Pendiente de documentar."),
        "Tipo_dato": str(serie.dtype),
        "Tipo_variable": (
            "Categórica" if str(serie.dtype) == "category"
            else "Numérica" if pd.api.types.is_numeric_dtype(serie)
            else "Texto / pendiente de conversión"
        ),
        "Valores_ejemplo": " | ".join(ejemplos),
        "Valores_faltantes": int(serie.isna().sum()),
        "Porcentaje_faltante": round(serie.isna().mean() * 100, 2),
        "Valores_unicos": int(serie.nunique()),
        "Uso_en_proyecto": usos.get(col, "Pendiente de definir."),
        "Observaciones": observaciones.get(col, "")
    })

diccionario = pd.DataFrame(registros_diccionario)

print(diccionario.to_string(index=False))

# Guardar diccionario

RUTA_DICCIONARIO = PROYECTO / "reports" / "diccionario_datos.csv"

diccionario.to_csv(
    RUTA_DICCIONARIO,
    index=False,
    encoding="utf-8-sig"
)

print("\nDiccionario generado correctamente:")
print(RUTA_DICCIONARIO)

# %% Exploración inicial
print("\nValores faltantes:")
print(df.isna().sum())
print("\nPorcentaje de valores faltantes:")
print((df.isna().mean() * 100).round(2))
print("\nEstadísticas descriptivas:")
print(df.describe(include="all").T)
print("\nIndicadores generales:")
print("Total de registros:", len(df))
print("Matrículas únicas:", df["MATRICULA"].nunique())
print("Actos jurídicos:", df["NOMBRE_NATUJUR"].nunique())
print("Municipios:", df["MUNICIPIO"].nunique())

print("\nIndicadores de valor:")
print("Registros con TIENE_VALOR=1:", df["TIENE_VALOR"].sum())
print(
    "Porcentaje sin valor:",
    round((1 - df["TIENE_VALOR"].mean()) * 100, 1)
)
print(
    "Registros de dinámica inmobiliaria:",
    df["Dinámica_Inmobiliaria"].sum()
)

print(df["DEPARTAMENTO"].value_counts())

# %% Revisar la variable VALOR

## 1. Conversión

df["VALOR_NUM"] = pd.to_numeric(
    df["VALOR"].str.replace(",", "", regex=False),
    errors="coerce"
)

# 2. Clasificación de los registros

df["ESTADO_VALOR"] = "POSITIVO"

df.loc[df["VALOR_NUM"].isna(), "ESTADO_VALOR"] = "NULO"
df.loc[df["VALOR_NUM"] == 0, "ESTADO_VALOR"] = "CERO"
df.loc[df["VALOR_NUM"] < 0, "ESTADO_VALOR"] = "NEGATIVO"

# 3. Distribución de estados

print("\nDistribución de estados de VALOR:")
print(df["ESTADO_VALOR"].value_counts(dropna=False))

# 4. Comparación con el indicador TIENE_VALOR

print("\nCruce entre TIENE_VALOR y ESTADO_VALOR:")

print(pd.crosstab(
    df["TIENE_VALOR"],
    df["ESTADO_VALOR"],
    margins=True
))

# 5. Calidad de los valores monetarios

con_valor = df[df["TIENE_VALOR"] == 1]

n0 = (con_valor["VALOR_NUM"] == 0).sum()

print("\nRegistros con TIENE_VALOR=1 y VALOR=0:")
print(n0, f"({n0 / len(con_valor) * 100:.1f}%)")

positivo = df["VALOR_NUM"] > 0

print(
    "\nPorcentaje de registros sin valor positivo:",
    round((1 - positivo.mean()) * 100, 1), "%"
)

# 6. Actos jurídicos asociados a valores en cero

print("\nPrincipales actos jurídicos con valor cero:")

print(
    con_valor.loc[
        con_valor["VALOR_NUM"] == 0,
        "NOMBRE_NATUJUR"
    ].value_counts().head(10)
)

# 7. Registros con múltiples valores

print("\nDistribución de TIENE_MAS_DE_UN_VALOR:")

print(df["TIENE_MAS_DE_UN_VALOR"].value_counts(dropna=False))

# 8. Estadísticas descriptivas de valores positivos

positivos = df.loc[df["VALOR_NUM"] > 0].copy()

print("\nEstadísticas descriptivas de VALOR:")
print(
    positivos["VALOR_NUM"].describe(
        percentiles=[
            0.01, 0.05, 0.25, 0.50, 0.75,
            0.90, 0.95, 0.99, 0.999
        ]
    )
)

# 9. Veinte valores monetarios más altos

print("\n20 valores monetarios más altos:")

print(
    positivos.nlargest(20, "VALOR_NUM")[
        [
            "MATRICULA",
            "YEAR_RADICA",
            "MUNICIPIO",
            "NOMBRE_NATUJUR",
            "VALOR",
            "VALOR_NUM"
        ]
    ].to_string(index=False)
)

# 10. Veinte valores monetarios positivos más bajos

print("\n20 valores monetarios positivos más bajos:")

print(
    positivos.nsmallest(20, "VALOR_NUM")[
        [
            "MATRICULA",
            "YEAR_RADICA",
            "MUNICIPIO",
            "NOMBRE_NATUJUR",
            "VALOR",
            "VALOR_NUM"
        ]
    ].to_string(index=False)
)

# 11. Percentiles de los valores positivos

print("\nPercentiles de VALOR:")

print(
    positivos["VALOR_NUM"].quantile(
        [
            0.01, 0.05, 0.25, 0.50, 0.75,
            0.90, 0.95, 0.99, 0.999
        ]
    )
)


# %% Dinámica anual

anual = df.groupby("YEAR_RADICA", observed=True).agg(
    registros=("TIENE_VALOR", "size"),
    prop_con_valor=("TIENE_VALOR", "mean"),
    prop_dinamica=("Dinámica_Inmobiliaria", "mean"),
    predios_nuevos=("PREDIOS_NUEVOS", "sum")
)

# Proporciones a porcentajes 
anual["prop_con_valor"] = anual["prop_con_valor"] * 100
anual["prop_dinamica"] = anual["prop_dinamica"] * 100

print("\nDinámica inmobiliaria anual:")
print(anual.round(2))

# Gráfico 1: registros por año
anual["registros"].plot(
    kind="bar",
    title="Número de registros por año",
    xlabel="Año de radicación",
    ylabel="Número de registros"
)
guardar("registros_anio")

# Gráfico 2: predios nuevos por año
anual["predios_nuevos"].plot(
    marker="o",
    title="Predios nuevos por año",
    xlabel="Año de radicación",
    ylabel="Número de predios nuevos"
)
guardar("predios_nuevos")

# Gráfico 3: porcentaje de registros con valor
anual["prop_con_valor"].plot(
    marker="o",
    title="Porcentaje de registros con valor por año",
    xlabel="Año de radicación",
    ylabel="Porcentaje (%)"
)
guardar("proporcion_valor_anio")

# %% Concentración de actos jurídicos

# Frecuencia de cada acto jurídico
actos = df["NOMBRE_NATUJUR"].value_counts()

# Concentración de los actos más frecuentes
print("\nConcentración de actos jurídicos:")
print(
    "Los 10 actos más frecuentes representan:",
    round(actos.head(10).sum() / len(df) * 100, 1),
    "%"
)

print(
    "Actos con menos de 1.000 registros:",
    (actos < 1000).sum(),
    "de",
    len(actos)
)

# Gráfico 1: 15 actos jurídicos más frecuentes
actos.head(15).sort_values().plot(
    kind="barh",
    title="15 actos jurídicos más frecuentes",
    xlabel="Número de registros",
    ylabel="Acto jurídico"
)
guardar("actos_top")

# Gráfico 2: porcentaje sin valor por acto jurídico
top = actos.head(15).index

sub = df[df["NOMBRE_NATUJUR"].isin(top)]

sin_valor_acto = (
    1 - sub.groupby("NOMBRE_NATUJUR", observed=True)["TIENE_VALOR"].mean()
) * 100

sin_valor_acto.sort_values().plot(
    kind="barh",
    title="Porcentaje sin valor en los 15 actos más frecuentes",
    xlabel="Registros sin valor (%)",
    ylabel="Acto jurídico"
)
guardar("sin_valor_acto")

# %% Heterogeneidad territorial
mun = df.groupby("MUNICIPIO", observed=True)["TIENE_VALOR"].agg(registros="size", con_valor="mean")
mun["pct_sin_valor"] = (1 - mun["con_valor"]) * 100
print("\nMunicipios con más registros:")
print(
    mun.sort_values("registros", ascending=False)
       .head(15)
       .round(2)
)
print("\nDistribución del porcentaje sin valor entre municipios:")
print(mun["pct_sin_valor"].describe().round(2))

print("\nMunicipios con mayor porcentaje sin valor:")
print(
    mun.sort_values("pct_sin_valor", ascending=False)
       .head(15)
       .round(2)
)

print("\nTotal de municipios:", len(mun))

# Gráfico 1: municipios con más registros
mun["registros"].nlargest(15).sort_values().plot(
    kind="barh",
    title="15 municipios con más registros",
    xlabel="Número de registros",
    ylabel="Municipio"
)
guardar("municipios_top")

# Gráfico 2: distribución del porcentaje sin valor
mun["pct_sin_valor"].plot(
    kind="hist",
    bins=20,
    title="Distribución del porcentaje sin valor por municipio",
    xlabel="Registros sin valor (%)",
    ylabel="Número de municipios"
)
guardar("sin_valor_municipio")

# %% Registros y diversidad de actos por matrícula

# Número de registros por matrícula
reg_mat = df.groupby("MATRICULA", observed=True).size()

print("\nEstadísticas de registros por matrícula:")
print(
    reg_mat.describe(
        percentiles=[0.50, 0.90, 0.99, 0.999]
    )
)

print("\n10 matrículas con más registros:")
print(reg_mat.sort_values(ascending=False).head(10))

print("\nNúmero de matrículas con más de un registro:")
print((reg_mat > 1).sum())

print(
    "Porcentaje de matrículas con más de un registro:",
    round((reg_mat > 1).mean() * 100, 2),
    "%"
)

# Gráfico: registros por matrícula hasta el percentil 99
limite = reg_mat.quantile(0.99)

reg_mat[reg_mat <= limite].plot(
    kind="hist",
    bins=50,
    title="Distribución de registros por matrícula (hasta p99)",
    xlabel="Número de registros por matrícula",
    ylabel="Número de matrículas"
)
guardar("registros_matricula")

# Diversidad de actos jurídicos por matrícula
actos_mat = df.groupby("MATRICULA", observed=True)["COD_NATUJUR"].nunique()

print("\nDiversidad de actos jurídicos por matrícula:")
print(
    actos_mat.describe(
        percentiles=[0.50, 0.90, 0.99]
    )
)

# %% Distribución del valor y valor por acto

# Seleccionar únicamente registros con valor positivo
pos = df.loc[df["VALOR_NUM"] > 0].copy()

# Transformación logarítmica 
pos["LOG_VALOR"] = np.log10(pos["VALOR_NUM"].astype("float64"))

# Estadísticas descriptivas
print("\nEstadísticas de VALOR para registros positivos:")
print(
    pos["VALOR_NUM"].describe(
        percentiles=[0.01, 0.50, 0.90, 0.95, 0.99, 0.999]
    )
)

print("\nEstadísticas de log10(VALOR):")
print(pos["LOG_VALOR"].describe().round(2))

# Gráfico 1: distribución logarítmica del valor
pos["LOG_VALOR"].plot(
    kind="hist",
    bins=60,
    title="Distribución logarítmica del valor monetario",
    xlabel="log10(VALOR)",
    ylabel="Frecuencia"
)
guardar("valor_log")

# Identificar los 8 actos jurídicos con más registros positivos
top_val = pos["NOMBRE_NATUJUR"].value_counts().head(8).index

box = pos[pos["NOMBRE_NATUJUR"].isin(top_val)].copy()

# Convertir a texto 
box["NOMBRE_NATUJUR"] = box["NOMBRE_NATUJUR"].astype(str)

# Gráfico 2: distribución del valor por acto jurídico
box.boxplot(
    column="LOG_VALOR",
    by="NOMBRE_NATUJUR",
    rot=45,
    figsize=(12, 7)
)

plt.suptitle("")
plt.title("Distribución de log10(VALOR) por acto jurídico")
plt.xlabel("Acto jurídico")
plt.ylabel("log10(VALOR)")
plt.tight_layout()

guardar("valor_por_acto")