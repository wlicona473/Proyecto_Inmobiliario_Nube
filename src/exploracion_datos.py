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

RUTA = Path("transacciones_bogota_cundinamarca.parquet")  
FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)

def guardar(nombre):
    plt.savefig(FIG / f"{nombre}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% Esquema 
pf = pq.ParquetFile(RUTA)
print(pf.metadata.num_rows, pf.metadata.num_columns)
print(pf.schema_arrow)

# %% Carga liviana
COLS = ["MATRICULA", "YEAR_RADICA", "DIVIPOLA", "DEPARTAMENTO", "MUNICIPIO",
        "TIPO_PREDIO_ZONA", "NUM_ANOTACION", "ESTADO_FOLIO", "Dinámica_Inmobiliaria",
        "COD_NATUJUR", "NOMBRE_NATUJUR", "PREDIOS_NUEVOS", "TIENE_VALOR",
        "TIENE_MAS_DE_UN_VALOR", "VALOR"]
CATS = ["MATRICULA", "DIVIPOLA", "DEPARTAMENTO", "MUNICIPIO", "TIPO_PREDIO_ZONA",
        "ESTADO_FOLIO", "COD_NATUJUR", "NOMBRE_NATUJUR"]
df = pq.read_table(RUTA, columns=COLS, read_dictionary=CATS).to_pandas()
df.info(memory_usage="deep")

# %% Validación de cifras del reporte 
print(len(df), df["MATRICULA"].nunique(), df["NOMBRE_NATUJUR"].nunique(), df["MUNICIPIO"].nunique())
print(df["TIENE_VALOR"].sum(), round((1 - df["TIENE_VALOR"].mean()) * 100, 1), df["Dinámica_Inmobiliaria"].sum())
print(df["DEPARTAMENTO"].value_counts())

# %% Conversión del valor
df["VALOR_NUM"] = pd.to_numeric(df["VALOR"].str.replace(",", "", regex=False), errors="coerce")
df = df.drop(columns="VALOR")
print(df["VALOR_NUM"].isna().sum())   

# %% Valores en cero y calidad
con_valor = df[df["TIENE_VALOR"] == 1]
n0 = (con_valor["VALOR_NUM"] == 0).sum()
print("TIENE_VALOR=1 con valor 0:", n0, f"({n0 / len(con_valor) * 100:.1f}%)")
positivo = df["VALOR_NUM"].fillna(0) > 0
print("% sin valor positivo sobre el total:", round((1 - positivo.mean()) * 100, 1))
print(con_valor.loc[con_valor["VALOR_NUM"] == 0, "NOMBRE_NATUJUR"].value_counts().head(10))
print(df["TIENE_MAS_DE_UN_VALOR"].value_counts(dropna=False))
print((df.isna().mean() * 100).round(2).sort_values(ascending=False))

# %% Dinámica anual
anual = df.groupby("YEAR_RADICA").agg(
    registros=("TIENE_VALOR", "size"),
    prop_con_valor=("TIENE_VALOR", "mean"),
    prop_dinamica=("Dinámica_Inmobiliaria", "mean"),
    predios_nuevos=("PREDIOS_NUEVOS", "sum"))
print(anual)
anual["registros"].plot(kind="bar", title="Registros por año"); guardar("registros_anio")
anual["predios_nuevos"].plot(marker="o", title="Predios nuevos por año"); guardar("predios_nuevos")

# %% Concentración de actos jurídicos
actos = df["NOMBRE_NATUJUR"].value_counts()
print("Top 10 actos concentran:", round(actos.head(10).sum() / len(df) * 100, 1), "%")
print("Actos con menos de 1.000 registros:", (actos < 1000).sum(), "de", len(actos))
actos.head(15).sort_values().plot(kind="barh", title="Actos jurídicos más frecuentes"); guardar("actos_top")

top = actos.head(15).index
sub = df[df["NOMBRE_NATUJUR"].isin(top)]
(1 - sub.groupby("NOMBRE_NATUJUR", observed=True)["TIENE_VALOR"].mean()).mul(100).sort_values() \
    .plot(kind="barh", title="% de registros sin valor por acto"); guardar("sin_valor_acto")

# %% Heterogeneidad territorial
mun = df.groupby("MUNICIPIO", observed=True)["TIENE_VALOR"].agg(registros="size", con_valor="mean")
mun["pct_sin_valor"] = (1 - mun["con_valor"]) * 100
print(mun.sort_values("registros", ascending=False).head(15))
print(mun["pct_sin_valor"].describe())
mun["registros"].sort_values(ascending=False).head(15).sort_values().plot(kind="barh", title="Municipios con más registros"); guardar("municipios_top")
mun["pct_sin_valor"].plot(kind="hist", bins=20, title="% sin valor por municipio"); guardar("sin_valor_municipio")

# %% Registros y diversidad de actos por matrícula
reg_mat = df.groupby("MATRICULA", observed=True).size()
print(reg_mat.describe(percentiles=[.5, .9, .99, .999]))
print(reg_mat.sort_values(ascending=False).head(10))
reg_mat[reg_mat <= reg_mat.quantile(.99)].plot(kind="hist", bins=50, title="Registros por matrícula (hasta p99)"); guardar("registros_matricula")
actos_mat = df.groupby("MATRICULA", observed=True)["COD_NATUJUR"].nunique()
print(actos_mat.describe(percentiles=[.5, .9, .99]))

# %% Distribución del valor y valor por acto
pos = df[positivo].copy()
pos["LOG_VALOR"] = np.log10(pos["VALOR_NUM"].astype("float64"))
print(pos["VALOR_NUM"].describe(percentiles=[.01, .5, .99, .999]))
pos["LOG_VALOR"].plot(kind="hist", bins=60, title="log10(valor), registros con valor positivo"); guardar("valor_log")

top_val = pos["NOMBRE_NATUJUR"].value_counts().head(8).index
box = pos[pos["NOMBRE_NATUJUR"].isin(top_val)].copy()
box["NOMBRE_NATUJUR"] = box["NOMBRE_NATUJUR"].astype(str)
box.boxplot(column="LOG_VALOR", by="NOMBRE_NATUJUR", rot=90, figsize=(10, 6))
plt.suptitle(""); plt.title("log10(valor) por acto (8 actos con más valores)"); guardar("valor_por_acto")