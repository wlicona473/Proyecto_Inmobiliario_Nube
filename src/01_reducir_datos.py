from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# 1. RUTAS DEL PROYECTO
# ============================================================

# Carpeta raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Archivo original
RUTA_ORIGINAL = (
    BASE_DIR
    / "data"
    / "raw"
    / "Registro_de_transacciones_inmobiliarias_en_Colombia_IGAC_20261016.csv"
)

# Archivo procesado
RUTA_PARQUET = (
    BASE_DIR
    / "data"
    / "processed"
    / "transacciones_bogota_cundinamarca.parquet"
)


# ============================================================
# 2. PARÁMETROS
# ============================================================

DEPARTAMENTOS_OBJETIVO = [
    "BOGOTÁ D.C.",
    "CUNDINAMARCA",
]

COLUMNAS_ENTERAS = [
    "YEAR_RADICA",
    "Dinámica_Inmobiliaria",
    "COUNT_A",
    "COUNT_DE",
    "PREDIOS_NUEVOS",
    "TIENE_VALOR",
    "TIENE_MAS_DE_UN_VALOR",
]

TAMANO_BLOQUE = 250_000


# ============================================================
# 3. PROCESAMIENTO
# ============================================================

def reducir_datos():
    """
    Lee el archivo nacional de transacciones inmobiliarias por bloques,
    conserva únicamente Bogotá D.C. y Cundinamarca y guarda el resultado
    en formato Parquet.
    """

    # Verificar que exista el archivo original
    if not RUTA_ORIGINAL.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo original:\n{RUTA_ORIGINAL}"
        )

    # Crear carpeta processed si no existe
    RUTA_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Eliminar una versión incompleta previa
    if RUTA_PARQUET.exists():
        RUTA_PARQUET.unlink()

    writer = None
    total_guardados = 0

    try:

        for i, bloque in enumerate(
            pd.read_csv(
                RUTA_ORIGINAL,
                chunksize=TAMANO_BLOQUE,
                dtype="string",
            ),
            start=1,
        ):

            # Filtrar únicamente Bogotá y Cundinamarca
            filtrado = bloque[
                bloque["DEPARTAMENTO"].isin(
                    DEPARTAMENTOS_OBJETIVO
                )
            ].copy()

            if filtrado.empty:
                print(
                    f"Bloque {i}: "
                    "0 registros seleccionados"
                )
                continue

            # Convertir columnas numéricas conocidas
            for columna in COLUMNAS_ENTERAS:
                filtrado[columna] = (
                    pd.to_numeric(
                        filtrado[columna],
                        errors="coerce",
                    )
                    .astype("Int64")
                )

            # Convertir DataFrame a tabla PyArrow
            tabla = pa.Table.from_pandas(
                filtrado,
                preserve_index=False,
            )

            # Crear el escritor con el esquema
            # del primer bloque válido
            if writer is None:
                writer = pq.ParquetWriter(
                    RUTA_PARQUET,
                    tabla.schema,
                    compression="snappy",
                )

            writer.write_table(tabla)

            total_guardados += len(filtrado)

            print(
                f"Bloque {i}: "
                f"{len(filtrado):,} registros | "
                f"Acumulado: {total_guardados:,}"
            )

    finally:

        if writer is not None:
            writer.close()

    print("\nProceso terminado")
    print(
        f"Total guardado: "
        f"{total_guardados:,}"
    )
    print(
        f"Archivo generado:\n"
        f"{RUTA_PARQUET}"
    )


# ============================================================
# 4. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    reducir_datos()