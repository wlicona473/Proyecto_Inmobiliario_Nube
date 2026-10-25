from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

ruta_original = (
    BASE_DIR
    / "data"
    / "raw"
    / "Registro_de_transacciones_inmobiliarias_en_Colombia_IGAC_20261016.csv"
)

ruta_salida = (
    BASE_DIR
    / "data"
    / "processed"
    / "transacciones_bogota_cundinamarca.parquet"
)


# Parámetros del procesamiento
departamentos_objetivo = [
    "BOGOTÁ D.C.",
    "CUNDINAMARCA",
]

columnas_enteras = [
    "YEAR_RADICA",
    "Dinámica_Inmobiliaria",
    "COUNT_A",
    "COUNT_DE",
    "PREDIOS_NUEVOS",
    "TIENE_VALOR",
    "TIENE_MAS_DE_UN_VALOR",
]

tamano_bloque = 250_000


def reducir_datos():
    """
    Crea una versión reducida de la base nacional con los registros
    correspondientes a Bogotá D.C. y Cundinamarca.

    El archivo se lee por bloques para evitar cargar toda la base
    en memoria y el resultado se guarda en formato Parquet.
    """

    if not ruta_original.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo original en:\n{ruta_original}"
        )

    # Crear la carpeta de salida en caso de que todavía no exista
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    # Si ya existe una versión anterior, se reemplaza
    if ruta_salida.exists():
        ruta_salida.unlink()

    escritor = None
    total_registros = 0

    try:
        bloques = pd.read_csv(
            ruta_original,
            chunksize=tamano_bloque,
            dtype="string",
        )

        for numero_bloque, bloque in enumerate(bloques, start=1):

            datos_filtrados = bloque[
                bloque["DEPARTAMENTO"].isin(departamentos_objetivo)
            ].copy()

            if datos_filtrados.empty:
                print(
                    f"Bloque {numero_bloque}: "
                    "sin registros de Bogotá o Cundinamarca"
                )
                continue

            # Algunas columnas se leen inicialmente como texto para mantener
            # un esquema consistente entre bloques. Aquí recuperamos su tipo entero.
            for columna in columnas_enteras:
                datos_filtrados[columna] = pd.to_numeric(
                    datos_filtrados[columna],
                    errors="coerce",
                ).astype("Int64")

            tabla = pa.Table.from_pandas(
                datos_filtrados,
                preserve_index=False,
            )

            # El esquema del primer bloque filtrado se utiliza para todo el archivo
            if escritor is None:
                escritor = pq.ParquetWriter(
                    ruta_salida,
                    tabla.schema,
                    compression="snappy",
                )

            escritor.write_table(tabla)

            total_registros += len(datos_filtrados)

            print(
                f"Bloque {numero_bloque}: "
                f"{len(datos_filtrados):,} registros guardados | "
                f"Total acumulado: {total_registros:,}"
            )

    finally:
        if escritor is not None:
            escritor.close()

    print("\nProcesamiento finalizado.")
    print(f"Total de registros guardados: {total_registros:,}")
    print(f"Archivo generado en:\n{ruta_salida}")


if __name__ == "__main__":
    reducir_datos()