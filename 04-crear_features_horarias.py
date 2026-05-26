import pandas as pd
import numpy as np

# =========================
# 1. CARGAR BASE ENRIQUECIDA
# =========================
df = pd.read_csv("base_horaria_enriquecida.csv")
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
df["fecha"] = pd.to_datetime(df["fecha"])

target = "demanda_horaria"

# =========================
# 2. VARIABLES DE CALENDARIO QUE SÍ SE PUEDEN USAR DIRECTAMENTE
# =========================
df["hora_sin"] = np.sin(2 * np.pi * df["hora_dia"] / 24)
df["hora_cos"] = np.cos(2 * np.pi * df["hora_dia"] / 24)

df["weekhour"] = df["dia_semana"] * 24 + df["hora_dia"]
df["weekhour_sin"] = np.sin(2 * np.pi * df["weekhour"] / 168)
df["weekhour_cos"] = np.cos(2 * np.pi * df["weekhour"] / 168)

calendar_cols = [
    "fecha_hora",
    "anio",
    "mes",
    "dia_mes",
    "hora_dia",
    "dia_semana",
    "es_fin_semana",
    "es_festivo",
    "hora_sin",
    "hora_cos",
    "weekhour_sin",
    "weekhour_cos"
]

# =========================
# 3. IDENTIFICAR FACTORES QUE DEBEN IR REZAGADOS
# =========================
raw_factor_cols = [
    c for c in df.columns
    if c not in ["fecha_hora", "fecha", target, "weekhour"] + calendar_cols[1:]
]

print("Número de factores contemporáneos a rezagar:", len(raw_factor_cols))
print(raw_factor_cols)

# =========================
# 4. LAGS DE LA DEMANDA HORARIA
# =========================
demand_lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]

for lag in demand_lags:
    df[f"{target}_lag_{lag}"] = df[target].shift(lag)

# =========================
# 5. MEDIAS Y DESVIACIONES MÓVILES DE LA DEMANDA
# =========================
rolling_windows = [3, 6, 12, 24, 48, 72, 168]

for w in rolling_windows:
    shifted = df[target].shift(1)
    df[f"{target}_ma_{w}"] = shifted.rolling(w).mean()
    df[f"{target}_std_{w}"] = shifted.rolling(w).std()

# =========================
# 6. REZAGOS DE LOS FACTORES EXTRA
# =========================
factor_lags = [1, 24, 168]

for col in raw_factor_cols:
    for lag in factor_lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)

# =========================
# 7. CONSTRUIR BASE FINAL SIN FUGA DE INFORMACIÓN
# =========================
generated_cols = [
    c for c in df.columns
    if c not in ["fecha", "weekhour"] + raw_factor_cols
]

base_modelo = df[generated_cols].dropna().reset_index(drop=True)

# =========================
# 8. GUARDAR
# =========================
base_modelo.to_csv("base_modelo_horaria.csv", index=False)

print("\nPrimeras filas:")
print(base_modelo.head())

print("\nDimensión final:")
print(base_modelo.shape)

print("\nNúmero total de columnas:")
print(len(base_modelo.columns))

print("\nArchivo guardado: base_modelo_horaria.csv")