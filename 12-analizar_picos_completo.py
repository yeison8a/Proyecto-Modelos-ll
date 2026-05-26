import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. CARGAR BASE ENRIQUECIDA
# =========================
df = pd.read_csv("base_horaria_enriquecida.csv")
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
df["fecha"] = pd.to_datetime(df["fecha"])

print("Dimensión del archivo:", df.shape)
print("\nColumnas:")
print(df.columns.tolist())

# =========================
# 2. IDENTIFICAR COLUMNAS IMPORTANTES
# =========================
call_cols = [c for c in df.columns if c.startswith("call_")]
sev_cols = [c for c in df.columns if c.startswith("sev_")]
borough_cols = [c for c in df.columns if c.startswith("borough_")]
flag_cols = ["held_flag", "reopen_flag", "special_event_flag", "standby_flag", "transfer_flag"]
time_cols = ["avg_dispatch_resp", "avg_incident_resp", "avg_travel_time"]

# =========================
# 3. HORAS MÁS ALTAS
# =========================
top_picos = df.sort_values("demanda_horaria", ascending=False).head(20)

print("\n==============================")
print("TOP 20 HORAS CON MAYOR DEMANDA")
print("==============================")
print(top_picos[["fecha_hora", "demanda_horaria", "es_fin_semana", "es_festivo"]])

# =========================
# 4. PERCENTILES
# =========================
p95 = np.percentile(df["demanda_horaria"], 95)
p99 = np.percentile(df["demanda_horaria"], 99)

print("\nPercentil 95:", p95)
print("Percentil 99:", p99)

horas_pico = df[df["demanda_horaria"] >= p99].copy()

print("\nNúmero de horas por encima del percentil 99:", len(horas_pico))
print("\nPrimeras horas pico:")
print(horas_pico[["fecha_hora", "demanda_horaria"]].head(20))

# =========================
# 5. FUNCIÓN PARA EXPLICAR UNA HORA
# =========================
def explicar_hora(fecha_hora_objetivo):
    fila = df[df["fecha_hora"] == pd.to_datetime(fecha_hora_objetivo)]

    if fila.empty:
        print("\nNo se encontró esa fecha_hora:", fecha_hora_objetivo)
        return

    fila = fila.iloc[0]

    print("\n======================================")
    print("HORA ANALIZADA:", fila["fecha_hora"])
    print("Demanda horaria:", fila["demanda_horaria"])
    print("Fin de semana:", fila["es_fin_semana"])
    print("Festivo:", fila["es_festivo"])

    print("\n--- Flags operativos ---")
    for c in flag_cols:
        print(f"{c}: {fila[c]}")

    print("\n--- Tiempos promedio ---")
    for c in time_cols:
        print(f"{c}: {fila[c]:.2f}")

    print("\n--- Tipos de llamada dominantes ---")
    calls = pd.Series({c: fila[c] for c in call_cols}).sort_values(ascending=False)
    print(calls.head(10))

    print("\n--- Severidades dominantes ---")
    sevs = pd.Series({c: fila[c] for c in sev_cols}).sort_values(ascending=False)
    print(sevs.head(10))

    print("\n--- Boroughs dominantes ---")
    boroughs = pd.Series({c: fila[c] for c in borough_cols}).sort_values(ascending=False)
    print(boroughs.head(10))

# =========================
# 6. COMPARAR CONTEXTO DE UNA HORA
# =========================
def comparar_contexto(fecha_hora_objetivo):
    fh = pd.to_datetime(fecha_hora_objetivo)

    puntos = {
        "actual": fh,
        "hace_1_hora": fh - pd.Timedelta(hours=1),
        "hace_24_horas": fh - pd.Timedelta(hours=24),
        "hace_168_horas": fh - pd.Timedelta(hours=168),
    }

    filas = []
    for nombre, fecha_ref in puntos.items():
        temp = df[df["fecha_hora"] == fecha_ref]
        if not temp.empty:
            fila = temp.iloc[0]
            filas.append({
                "referencia": nombre,
                "fecha_hora": fila["fecha_hora"],
                "demanda_horaria": fila["demanda_horaria"],
                "es_fin_semana": fila["es_fin_semana"],
                "es_festivo": fila["es_festivo"],
                "special_event_flag": fila["special_event_flag"],
                "standby_flag": fila["standby_flag"],
                "transfer_flag": fila["transfer_flag"],
                "avg_dispatch_resp": fila["avg_dispatch_resp"],
                "avg_incident_resp": fila["avg_incident_resp"],
                "avg_travel_time": fila["avg_travel_time"],
            })

    comp = pd.DataFrame(filas)

    print("\n==============================")
    print("COMPARACIÓN DE CONTEXTO")
    print("==============================")
    print(comp)

# =========================
# 7. PERFIL PROMEDIO DE HORAS PICO
# =========================
print("\n==========================================")
print("PERFIL PROMEDIO DE HORAS PICO (>= P99)")
print("==========================================")
print("Promedio de demanda en horas pico:", horas_pico["demanda_horaria"].mean())

print("\nTipos de llamada promedio en horas pico:")
print(horas_pico[call_cols].mean().sort_values(ascending=False).head(10))

print("\nSeveridades promedio en horas pico:")
print(horas_pico[sev_cols].mean().sort_values(ascending=False).head(10))

print("\nBoroughs promedio en horas pico:")
print(horas_pico[borough_cols].mean().sort_values(ascending=False).head(10))

print("\nPromedio de flags en horas pico:")
print(horas_pico[flag_cols].mean().sort_values(ascending=False))

print("\nPromedio de tiempos en horas pico:")
print(horas_pico[time_cols].mean())

# =========================
# 8. COMPARAR PICO VS NORMAL
# =========================
pico = df[df["demanda_horaria"] >= p99].copy()
normal = df[df["demanda_horaria"] < p99].copy()

comparacion = pd.DataFrame({
    "pico": pico[call_cols + sev_cols + borough_cols + flag_cols + time_cols].mean(),
    "normal": normal[call_cols + sev_cols + borough_cols + flag_cols + time_cols].mean()
})

comparacion["diferencia"] = comparacion["pico"] - comparacion["normal"]

print("\n===================================================")
print("VARIABLES CON MAYOR DIFERENCIA ENTRE PICO Y NORMAL")
print("===================================================")
print(comparacion.sort_values("diferencia", ascending=False).head(25))

# =========================
# 9. GRÁFICA DE TODAS LAS HORAS PICO
# =========================
plt.figure(figsize=(16, 6))
plt.plot(df["fecha_hora"], df["demanda_horaria"], label="Demanda horaria")
plt.scatter(horas_pico["fecha_hora"], horas_pico["demanda_horaria"], label="Horas pico (>= P99)")
plt.title("Horas pico en la serie")
plt.xlabel("Fecha")
plt.ylabel("Demanda horaria")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 10. GRAFICAR TOP 20 PICOS
# =========================
top20 = top_picos.sort_values("fecha_hora")

plt.figure(figsize=(16, 6))
plt.bar(top20["fecha_hora"].astype(str), top20["demanda_horaria"])
plt.title("Top 20 horas con mayor demanda")
plt.xlabel("Fecha-hora")
plt.ylabel("Demanda horaria")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# =========================
# 11. EJEMPLOS AUTOMÁTICOS
# =========================
print("\n======================================")
print("ANÁLISIS AUTOMÁTICO DE LAS 3 HORAS MÁS ALTAS")
print("======================================")

top3_fechas = top_picos["fecha_hora"].head(3).tolist()

for fh in top3_fechas:
    explicar_hora(fh)
    comparar_contexto(fh)