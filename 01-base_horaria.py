import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar

# =========================
# CONFIGURACIÓN
# =========================
archivo = "ems-incident-dispatch-data.csv"  
date_col = "INCIDENT_DATETIME"
chunksize = 500000

# =========================
# 1. CONTAR INCIDENTES POR FECHA-HORA
# =========================
conteo_total = pd.Series(dtype="int64")

for i, chunk in enumerate(pd.read_csv(
    archivo,
    usecols=[date_col],
    chunksize=chunksize
)):
    print(f"Procesando bloque {i+1}...")

    # convertir a datetime
    chunk[date_col] = pd.to_datetime(
        chunk[date_col],
        errors="coerce",
        format="mixed"
    )

    # quitar fechas inválidas
    chunk = chunk.dropna(subset=[date_col]).copy()

    # bajar a nivel de hora
    chunk["fecha_hora"] = chunk[date_col].dt.floor("h")

    # contar cuántos incidentes hubo en cada fecha-hora
    conteo_chunk = chunk["fecha_hora"].value_counts().sort_index()

    # acumular
    conteo_total = conteo_total.add(conteo_chunk, fill_value=0)

# convertir a DataFrame
hourly = conteo_total.sort_index().reset_index()
hourly.columns = ["fecha_hora", "demanda_horaria"]
hourly["demanda_horaria"] = hourly["demanda_horaria"].astype(int)

# =========================
# 2. COMPLETAR TODAS LAS 24 HORAS DE TODOS LOS DÍAS
# =========================
full_hours = pd.DataFrame({
    "fecha_hora": pd.date_range(
        start=hourly["fecha_hora"].min(),
        end=hourly["fecha_hora"].max(),
        freq="h"
    )
})

hourly = full_hours.merge(hourly, on="fecha_hora", how="left")
hourly["demanda_horaria"] = hourly["demanda_horaria"].fillna(0).astype(int)

# =========================
# 3. VARIABLES DE CALENDARIO
# =========================
hourly["anio"] = hourly["fecha_hora"].dt.year
hourly["mes"] = hourly["fecha_hora"].dt.month
hourly["dia_mes"] = hourly["fecha_hora"].dt.day
hourly["hora_dia"] = hourly["fecha_hora"].dt.hour
hourly["dia_semana"] = hourly["fecha_hora"].dt.dayofweek   # lunes=0
hourly["es_fin_semana"] = hourly["dia_semana"].isin([5, 6]).astype(int)

# =========================
# 4. FESTIVOS
# =========================
cal = USFederalHolidayCalendar()
holidays = cal.holidays(
    start=hourly["fecha_hora"].min().normalize(),
    end=hourly["fecha_hora"].max().normalize()
)

hourly["fecha"] = hourly["fecha_hora"].dt.normalize()
hourly["es_festivo"] = hourly["fecha"].isin(holidays).astype(int)

# =========================
# 5. GUARDAR RESULTADO
# =========================
hourly.to_csv("base_horaria.csv", index=False)

print("\nPrimeras filas:")
print(hourly.head())

print("\nÚltimas filas:")
print(hourly.tail())

print("\nDimensión final:")
print(hourly.shape)

print("\nRango temporal:")
print(hourly["fecha_hora"].min(), "->", hourly["fecha_hora"].max())

print("\nArchivo guardado: base_horaria.csv")