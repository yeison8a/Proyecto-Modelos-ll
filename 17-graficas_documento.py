import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
import os

# =========================
# CONFIG
# =========================
os.makedirs("resultados", exist_ok=True)

# =========================
# 1. CARGAR BASE
# =========================
df = pd.read_csv("base_modelo_horaria.csv")
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

target = "demanda_horaria"
features = [c for c in df.columns if c not in ["fecha_hora", target]]

# =========================
# 2. SPLIT
# =========================
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train = df.iloc[:train_end].copy()
val = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

X_train = train[features]
y_train = train[target]

X_val = val[features]
y_val = val[target]

X_test = test[features]
y_test = test[target]

# =========================
# 3. ENTRENAR HGBR
# =========================
hgbr = HistGradientBoostingRegressor(
    learning_rate=0.03,
    max_iter=800,
    max_depth=10,
    min_samples_leaf=30,
    l2_regularization=0.1,
    random_state=42
)
hgbr.fit(X_train, y_train)

pred_val_hgbr = hgbr.predict(X_val)
pred_test_hgbr = hgbr.predict(X_test)

# =========================
# 4. ENTRENAR XGB
# =========================
xgb = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=800,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

pred_val_xgb = xgb.predict(X_val)
pred_test_xgb = xgb.predict(X_test)

# =========================
# 5. ENSAMBLE FINAL
# =========================
alpha = 0.25
pred_ens_test = alpha * pred_test_hgbr + (1 - alpha) * pred_test_xgb

# guardar predicciones
predicciones = pd.DataFrame({
    "fecha_hora": test["fecha_hora"].values,
    "real": y_test.values,
    "prediccion": pred_ens_test
})
predicciones.to_csv("resultados/predicciones_ensamble_test.csv", index=False)

# =========================
# 6. MÉTRICAS
# =========================
mae = mean_absolute_error(y_test, pred_ens_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_ens_test))
wmape = np.sum(np.abs(y_test - pred_ens_test)) / np.sum(np.abs(y_test)) * 100

tabla = pd.DataFrame([{
    "Modelo": "Ensamble HGBR + XGBoost",
    "MAE": mae,
    "RMSE": rmse,
    "wMAPE": wmape
}])
tabla.to_csv("resultados/tabla_final_modelos.csv", index=False)

print(tabla)

# =========================
# 7. GRAFICA COMPLETA TEST
# =========================
plt.figure(figsize=(16, 6))
plt.plot(predicciones["fecha_hora"], predicciones["real"], label="Real", linewidth=1)
plt.plot(predicciones["fecha_hora"], predicciones["prediccion"], label="Predicción", linewidth=1)
plt.title("Demanda horaria real vs predicción - Ensamble")
plt.xlabel("Fecha")
plt.ylabel("Demanda horaria")
plt.legend()
plt.tight_layout()
plt.savefig("resultados/grafica_test_completa.png", dpi=200)
plt.show()

# =========================
# 8. ZOOM DE UNA SEMANA
# =========================
zoom = predicciones.iloc[:24*7]

plt.figure(figsize=(16, 6))
plt.plot(zoom["fecha_hora"], zoom["real"], label="Real", marker="o", markersize=2, linewidth=1)
plt.plot(zoom["fecha_hora"], zoom["prediccion"], label="Predicción", marker="o", markersize=2, linewidth=1)
plt.title("Zoom de una semana - Ensamble")
plt.xlabel("Fecha")
plt.ylabel("Demanda horaria")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("resultados/grafica_zoom_semana.png", dpi=200)
plt.show()

# =========================
# 9. PROMEDIO DIARIO
# =========================
predicciones["fecha"] = pd.to_datetime(predicciones["fecha_hora"]).dt.date
resumen_diario = predicciones.groupby("fecha")[["real", "prediccion"]].mean().reset_index()

plt.figure(figsize=(16, 6))
plt.plot(resumen_diario["fecha"], resumen_diario["real"], label="Real diario promedio", linewidth=2)
plt.plot(resumen_diario["fecha"], resumen_diario["prediccion"], label="Predicción diaria promedio", linewidth=2)
plt.title("Promedio diario de demanda horaria - Ensamble")
plt.xlabel("Fecha")
plt.ylabel("Demanda horaria promedio")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("resultados/grafica_promedio_diario.png", dpi=200)
plt.show()

# =========================
# 10. HISTOGRAMA DE ERRORES
# =========================
predicciones["error"] = predicciones["real"] - predicciones["prediccion"]

plt.figure(figsize=(10, 6))
plt.hist(predicciones["error"], bins=60)
plt.title("Distribución de errores del ensamble")
plt.xlabel("Error (real - predicción)")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("resultados/grafica_residuos.png", dpi=200)
plt.show()

# =========================
# 11. TOP PICOS
# =========================
top_picos = predicciones.sort_values("real", ascending=False).head(20).sort_values("fecha_hora")

plt.figure(figsize=(16, 6))
plt.bar(top_picos["fecha_hora"].astype(str), top_picos["real"], label="Real")
plt.plot(top_picos["fecha_hora"].astype(str), top_picos["prediccion"], marker="o", label="Predicción")
plt.title("Top 20 horas pico: real vs predicción")
plt.xlabel("Fecha-hora")
plt.ylabel("Demanda horaria")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig("resultados/grafica_top_picos.png", dpi=200)
plt.show()

print("\nGráficas guardadas en la carpeta 'resultados/'")