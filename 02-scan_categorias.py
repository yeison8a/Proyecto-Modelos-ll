import pandas as pd
from collections import Counter

# =========================
# CONFIGURACIÓN
# =========================
archivo = "ems-incident-dispatch-data.csv"
chunksize = 500000

# contadores globales
call_counter = Counter()
severity_counter = Counter()
borough_counter = Counter()

# columnas necesarias
usecols = [
    "INITIAL_CALL_TYPE",
    "INITIAL_SEVERITY_LEVEL_CODE",
    "BOROUGH"
]

# =========================
# FUNCIÓN DE LIMPIEZA
# =========================
def limpiar_texto(serie):
    """
    Limpia texto:
    - reemplaza NaN por UNKNOWN
    - convierte a string
    - elimina espacios
    - reemplaza vacíos por UNKNOWN
    """
    return (
        serie.fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .replace("", "UNKNOWN")
    )

# =========================
# LECTURA POR BLOQUES
# =========================
for i, chunk in enumerate(pd.read_csv(
    archivo,
    usecols=usecols,
    chunksize=chunksize,
    low_memory=False
)):
    print(f"Procesando bloque {i+1}...")

    # limpiar variables categóricas
    calls = limpiar_texto(chunk["INITIAL_CALL_TYPE"])
    sev = limpiar_texto(chunk["INITIAL_SEVERITY_LEVEL_CODE"])
    borough = limpiar_texto(chunk["BOROUGH"])

    # contar frecuencias del bloque actual
    call_counter.update(calls.value_counts().to_dict())
    severity_counter.update(sev.value_counts().to_dict())
    borough_counter.update(borough.value_counts().to_dict())

# =========================
# MOSTRAR RESULTADOS
# =========================

# top 10 tipos de llamada
print("\nTop 10 INITIAL_CALL_TYPE:")
for k, v in call_counter.most_common(10):
    print(f"{k}: {v}")

# distribución de severidad
print("\nValores de INITIAL_SEVERITY_LEVEL_CODE:")
for k, v in severity_counter.most_common():
    print(f"{k}: {v}")

# distribución por borough
print("\nValores de BOROUGH:")
for k, v in borough_counter.most_common():
    print(f"{k}: {v}")