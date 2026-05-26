import pandas as pd
import numpy as np
import re

# =========================
# CONFIGURACIÓN
# =========================
archivo = "ems-incident-dispatch-data.csv" 
chunksize = 500000

TOP_CALL_TYPES = [
    "SICK", "INJURY", "DIFFBR", "EDP", "DRUG",
    "UNC", "UNKNOW", "ABDPN", "CARD", "MVAINJ"
]

SEVERITY_LEVELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

BOROUGHS = [
    "BROOKLYN",
    "MANHATTAN",
    "BRONX",
    "QUEENS",
    "RICHMOND / STATEN ISLAND",
    "UNKNOWN"
]

usecols = [
    "INCIDENT_DATETIME",
    "INITIAL_CALL_TYPE",
    "INITIAL_SEVERITY_LEVEL_CODE",
    "BOROUGH",
    "VALID_DISPATCH_RSPNS_TIME_INDC",
    "DISPATCH_RESPONSE_SECONDS_QY",
    "VALID_INCIDENT_RSPNS_TIME_INDC",
    "INCIDENT_RESPONSE_SECONDS_QY",
    "INCIDENT_TRAVEL_TM_SECONDS_QY",
    "HELD_INDICATOR",
    "REOPEN_INDICATOR",
    "SPECIAL_EVENT_INDICATOR",
    "STANDBY_INDICATOR",
    "TRANSFER_INDICATOR"
]

# =========================
# FUNCIONES AUXILIARES
# =========================
def limpiar_texto(serie):
    return (
        serie.fillna("UNKNOWN")
             .astype(str)
             .str.strip()
             .replace("", "UNKNOWN")
    )

def flag_a_int(serie):
    s = limpiar_texto(serie).str.upper()
    return s.isin(["Y", "YES", "TRUE", "1", "T"]).astype(int)

def safe_name(texto):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(texto)).strip("_").upper()

call_cols = [f"call_{safe_name(x)}" for x in TOP_CALL_TYPES] + ["call_OTHER"]
sev_cols = [f"sev_{safe_name(x)}" for x in SEVERITY_LEVELS]
borough_cols = [f"borough_{safe_name(x)}" for x in BOROUGHS]

partes = []

# =========================
# 1. LEER EL ARCHIVO ORIGINAL POR BLOQUES
# =========================
for i, chunk in enumerate(pd.read_csv(
    archivo,
    usecols=usecols,
    chunksize=chunksize,
    low_memory=False
)):
    print(f"Procesando bloque {i+1}...")

    # -------------------------
    # FECHA-HORA
    # -------------------------
    chunk["INCIDENT_DATETIME"] = pd.to_datetime(
        chunk["INCIDENT_DATETIME"],
        errors="coerce",
        format="mixed"
    )
    chunk = chunk.dropna(subset=["INCIDENT_DATETIME"]).copy()
    chunk["fecha_hora"] = chunk["INCIDENT_DATETIME"].dt.floor("h")

    # -------------------------
    # LIMPIEZA DE CATEGORÍAS
    # -------------------------
    call_type = limpiar_texto(chunk["INITIAL_CALL_TYPE"])
    sev = limpiar_texto(chunk["INITIAL_SEVERITY_LEVEL_CODE"])
    borough = limpiar_texto(chunk["BOROUGH"])

    # agrupar tipos de llamada
    call_type = call_type.where(call_type.isin(TOP_CALL_TYPES), "OTHER")

    # si llega alguna severidad rara, la dejamos como UNKNOWN_RARE
    sev = sev.where(sev.isin(SEVERITY_LEVELS), "UNKNOWN_RARE")

    # si llega algún borough raro, lo dejamos como UNKNOWN
    borough = borough.where(borough.isin(BOROUGHS), "UNKNOWN")

    # -------------------------
    # DUMMIES DE TIPO DE LLAMADA
    # -------------------------
    call_dummies = pd.get_dummies(call_type)
    for c in TOP_CALL_TYPES + ["OTHER"]:
        if c not in call_dummies.columns:
            call_dummies[c] = 0
    call_dummies = call_dummies[TOP_CALL_TYPES + ["OTHER"]]
    call_dummies.columns = call_cols

    # -------------------------
    # DUMMIES DE SEVERIDAD
    # -------------------------
    sev_dummies = pd.get_dummies(sev)
    for s in SEVERITY_LEVELS:
        if s not in sev_dummies.columns:
            sev_dummies[s] = 0
    sev_dummies = sev_dummies[SEVERITY_LEVELS]
    sev_dummies.columns = sev_cols

    # -------------------------
    # DUMMIES DE BOROUGH
    # -------------------------
    borough_dummies = pd.get_dummies(borough)
    for b in BOROUGHS:
        if b not in borough_dummies.columns:
            borough_dummies[b] = 0
    borough_dummies = borough_dummies[BOROUGHS]
    borough_dummies.columns = borough_cols

    # -------------------------
    # FLAGS OPERATIVOS
    # -------------------------
    chunk["held_flag"] = flag_a_int(chunk["HELD_INDICATOR"])
    chunk["reopen_flag"] = flag_a_int(chunk["REOPEN_INDICATOR"])
    chunk["special_event_flag"] = flag_a_int(chunk["SPECIAL_EVENT_INDICATOR"])
    chunk["standby_flag"] = flag_a_int(chunk["STANDBY_INDICATOR"])
    chunk["transfer_flag"] = flag_a_int(chunk["TRANSFER_INDICATOR"])

    # -------------------------
    # TIEMPOS DE RESPUESTA
    # -------------------------
    chunk["valid_dispatch_flag"] = flag_a_int(chunk["VALID_DISPATCH_RSPNS_TIME_INDC"])
    chunk["valid_incident_flag"] = flag_a_int(chunk["VALID_INCIDENT_RSPNS_TIME_INDC"])

    for col in [
        "DISPATCH_RESPONSE_SECONDS_QY",
        "INCIDENT_RESPONSE_SECONDS_QY",
        "INCIDENT_TRAVEL_TM_SECONDS_QY"
    ]:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    # solo usar tiempos válidos cuando aplique
    chunk["dispatch_resp_valid"] = np.where(
        (chunk["valid_dispatch_flag"] == 1) & (chunk["DISPATCH_RESPONSE_SECONDS_QY"].notna()),
        chunk["DISPATCH_RESPONSE_SECONDS_QY"],
        np.nan
    )

    chunk["incident_resp_valid"] = np.where(
        (chunk["valid_incident_flag"] == 1) & (chunk["INCIDENT_RESPONSE_SECONDS_QY"].notna()),
        chunk["INCIDENT_RESPONSE_SECONDS_QY"],
        np.nan
    )

    chunk["travel_time_valid"] = chunk["INCIDENT_TRAVEL_TM_SECONDS_QY"]

    # -------------------------
    # BLOQUE DE CONTEOS
    # -------------------------
    base_conteos = pd.concat([
        chunk[[
            "fecha_hora",
            "held_flag",
            "reopen_flag",
            "special_event_flag",
            "standby_flag",
            "transfer_flag"
        ]],
        call_dummies,
        sev_dummies,
        borough_dummies
    ], axis=1)

    agg_conteos = base_conteos.groupby("fecha_hora").sum()

    # -------------------------
    # BLOQUE DE PROMEDIOS
    # -------------------------
    aux = pd.DataFrame({
        "fecha_hora": chunk["fecha_hora"],

        "dispatch_resp_sum": pd.Series(chunk["dispatch_resp_valid"]).fillna(0),
        "dispatch_resp_count": pd.Series(chunk["dispatch_resp_valid"]).notna().astype(int),

        "incident_resp_sum": pd.Series(chunk["incident_resp_valid"]).fillna(0),
        "incident_resp_count": pd.Series(chunk["incident_resp_valid"]).notna().astype(int),

        "travel_sum": pd.Series(chunk["travel_time_valid"]).fillna(0),
        "travel_count": pd.Series(chunk["travel_time_valid"]).notna().astype(int),
    })

    agg_aux = aux.groupby("fecha_hora").sum()

    # unir bloque del chunk
    chunk_agg = agg_conteos.join(agg_aux, how="outer").fillna(0)
    partes.append(chunk_agg)

# =========================
# 2. UNIR TODOS LOS BLOQUES
# =========================
agg_total = pd.concat(partes).groupby(level=0).sum().sort_index()

# =========================
# 3. CALCULAR PROMEDIOS FINALES
# =========================
agg_total["avg_dispatch_resp"] = agg_total["dispatch_resp_sum"] / agg_total["dispatch_resp_count"].replace(0, np.nan)
agg_total["avg_incident_resp"] = agg_total["incident_resp_sum"] / agg_total["incident_resp_count"].replace(0, np.nan)
agg_total["avg_travel_time"] = agg_total["travel_sum"] / agg_total["travel_count"].replace(0, np.nan)

agg_total["avg_dispatch_resp"] = agg_total["avg_dispatch_resp"].fillna(0)
agg_total["avg_incident_resp"] = agg_total["avg_incident_resp"].fillna(0)
agg_total["avg_travel_time"] = agg_total["avg_travel_time"].fillna(0)

agg_total = agg_total.drop(columns=[
    "dispatch_resp_sum", "dispatch_resp_count",
    "incident_resp_sum", "incident_resp_count",
    "travel_sum", "travel_count"
])

agg_total = agg_total.reset_index()

# =========================
# 4. CARGAR LA BASE HORARIA Y HACER EL MERGE
# =========================
base_horaria = pd.read_csv("base_horaria.csv")
base_horaria["fecha_hora"] = pd.to_datetime(base_horaria["fecha_hora"])
base_horaria["fecha"] = pd.to_datetime(base_horaria["fecha"])

base_final = base_horaria.merge(
    agg_total,
    on="fecha_hora",
    how="left"
)

# rellenar columnas nuevas faltantes con 0
nuevas_cols = [c for c in base_final.columns if c not in base_horaria.columns]
base_final[nuevas_cols] = base_final[nuevas_cols].fillna(0)

# =========================
# 5. GUARDAR
# =========================
base_final.to_csv("base_horaria_enriquecida.csv", index=False)

print("\nPrimeras filas:")
print(base_final.head())

print("\nDimensión final:")
print(base_final.shape)

print("\nColumnas nuevas agregadas:")
print(nuevas_cols)

print("\nArchivo guardado: base_horaria_enriquecida.csv")