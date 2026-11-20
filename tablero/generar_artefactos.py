"""
Genera catalogos.json: nombres legibles de municipios y actos jurídicos para los
desplegables del formulario (el modelo solo conoce los códigos DIVIPOLA y COD_NATUJUR).

Ejecutar una vez desde la carpeta tablero/:
    .venv\\Scripts\\python.exe generar_artefactos.py --parquet C:\\proyecto\\datos\\transacciones_bogota_cundinamarca.parquet

La importancia global de las variables ya no necesita este script: el tablero la calcula
directamente desde el modelo (prediccion.py).
"""
import argparse
import json

import numpy as np
import pandas as pd

COL_MUNICIPIO = "MUNICIPIO"
COL_ACTO = "NOMBRE_NATUJUR"


def _s(v):
    """Código como texto limpio: 11001.0 -> '11001'."""
    if pd.isna(v):
        return v
    if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)


def main(parquet):
    df = pd.read_parquet(parquet, columns=["DIVIPOLA", COL_MUNICIPIO, "COD_NATUJUR", COL_ACTO])
    df["DIVIPOLA"] = df["DIVIPOLA"].map(_s)
    df["COD_NATUJUR"] = df["COD_NATUJUR"].map(_s)

    cat = {}
    for var, col in (("DIVIPOLA", COL_MUNICIPIO), ("COD_NATUJUR", COL_ACTO)):
        u = df.dropna(subset=[var, col]).drop_duplicates(var)
        cat[var] = {k: str(v).title() for k, v in zip(u[var], u[col])}

    with open("catalogos.json", "w", encoding="utf-8") as f:
        json.dump(cat, f, ensure_ascii=False, indent=1)
    print(f"Municipios: {len(cat['DIVIPOLA'])} | Actos jurídicos: {len(cat['COD_NATUJUR'])}")
    print("Ejemplos:", list(cat["DIVIPOLA"].items())[:3], list(cat["COD_NATUJUR"].items())[:3])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    main(ap.parse_args().parquet)
