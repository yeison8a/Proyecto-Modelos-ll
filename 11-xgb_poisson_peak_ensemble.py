import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# =========================
# 1. CARGAR BASE FINAL HORARIA
# =========================
df = pd.read_csv("base_modelo_horaria.csv")
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

target = "demanda_horaria"
features = [c for c in df.columns if c not in ["fecha_hora", target]]

# =========================
# 2. SPLIT TEMPORAL
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
# 3. FUNCIÓN DE EVALUACIÓN
# =========================
def evaluar(y_real, y_pred, nombre="modelo"):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    wmape = np.sum(np.abs(y_real - y_pred)) / np.sum(np.abs(y_real)) * 100

    return {
        "modelo": nombre,
        "MAE": mae,
        "RMSE": rmse,
        "wMAPE": wmape
    }

# =========================
# 4. ENTRENAR HGBR
# =========================
print("Entrenando HGBR...")

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

print("HGBR validación:", evaluar(y_val, pred_val_hgbr, "HGBR_val"))
print("HGBR test:", evaluar(y_test, pred_test_hgbr, "HGBR_test"))

# =========================
# 5. ENTRENAR XGBOOST
# =========================
print("\nEntrenando XGBoost...")

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

print("XGBoost validación:", evaluar(y_val, pred_val_xgb, "XGB_val"))
print("XGBoost test:", evaluar(y_test, pred_test_xgb, "XGB_test"))

# =========================
# 6. BUSCAR MEJOR ALPHA EN VALIDACIÓN
# =========================
alphas = np.arange(0.0, 1.01, 0.05)

resultados_alpha = []
mejor_alpha = None
mejor_mae = float("inf")

for alpha in alphas:
    pred_ens_val = alpha * pred_val_hgbr + (1 - alpha) * pred_val_xgb
    res = evaluar(y_val, pred_ens_val, f"Ensamble_alpha_{alpha:.2f}")

    resultados_alpha.append({
        "alpha": alpha,
        "Val_MAE": res["MAE"],
        "Val_RMSE": res["RMSE"],
        "Val_wMAPE": res["wMAPE"]
    })

    if res["MAE"] < mejor_mae:
        mejor_mae = res["MAE"]
        mejor_alpha = alpha

tabla_alpha = pd.DataFrame(resultados_alpha).sort_values("Val_MAE")

print("\nTabla de alphas en validación:")
print(tabla_alpha)

print("\nMejor alpha encontrado:", mejor_alpha)

# =========================
# 7. EVALUAR ENSAMBLE EN TEST
# =========================
pred_ens_test = mejor_alpha * pred_test_hgbr + (1 - mejor_alpha) * pred_test_xgb

res_test = evaluar(y_test, pred_ens_test, f"Ensamble_test_alpha_{mejor_alpha:.2f}")

print("\nResultado final del ensamble en test:")
print(res_test)

# =========================
# 8. GUARDAR PREDICCIONES
# =========================
predicciones_test = pd.DataFrame({
    "fecha_hora": test["fecha_hora"].values,
    "real": y_test.values,
    "prediccion": pred_ens_test
})

predicciones_test.to_csv("predicciones_ensamble_test.csv", index=False)
print("\nArchivo guardado: predicciones_ensamble_test.csv")

# =========================
# 9. GRÁFICA COMPLETA
# =========================
plt.figure(figsize=(16, 6))
plt.plot(predicciones_test["fecha_hora"], predicciones_test["real"], label="Real", linewidth=1)
plt.plot(predicciones_test["fecha_hora"], predicciones_test["prediccion"], label="Predicción", linewidth=1)
plt.title("Demanda horaria real vs predicción - Ensamble HGBR + XGBoost")
plt.xlabel("Fecha")
plt.ylabel("Demanda horaria")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 10. ZOOM DE UNA SEMANA
# =========================
zoom = predicciones_test.iloc[:24*7]   # primeras 168 horas del test

plt.figure(figsize=(16, 6))
plt.plot(zoom["fecha_hora"], zoom["real"], label="Real", marker="o", markersize=2, linewidth=1)
plt.plot(zoom["fecha_hora"], zoom["prediccion"], label="Predicción", marker="o", markersize=2, linewidth=1)
plt.title("Zoom de una semana - Ensamble HGBR + XGBoost")
plt.xlabel("Fecha")
plt.ylabel("Demanda horaria")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =========================
# 11. PROMEDIO DIARIO
# =========================
predicciones_test["fecha"] = pd.to_datetime(predicciones_test["fecha_hora"]).dt.date

resumen_diario = predicciones_test.groupby("fecha")[["real", "prediccion"]].mean().reset_index()

plt.figure(figsize=(16, 6))
plt.plot(resumen_diario["fecha"], resumen_diario["real"], label="Real diario promedio", linewidth=2)
plt.plot(resumen_diario["fecha"], resumen_diario["prediccion"], label="Predicción diaria promedio", linewidth=2)
plt.title("Promedio diario de demanda horaria - Ensamble")
plt.xlabel("Fecha")
plt.ylabel("Demanda horaria promedio")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()