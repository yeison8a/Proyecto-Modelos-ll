import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

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
# 4. CONFIGURACIONES A PROBAR
# =========================
configs = [
    {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_depth": 6,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0
    },
    {
        "learning_rate": 0.05,
        "max_iter": 500,
        "max_depth": 8,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0
    },
    {
        "learning_rate": 0.03,
        "max_iter": 600,
        "max_depth": 8,
        "min_samples_leaf": 30,
        "l2_regularization": 0.1
    },
    {
        "learning_rate": 0.03,
        "max_iter": 800,
        "max_depth": 10,
        "min_samples_leaf": 30,
        "l2_regularization": 0.1
    }
]

resultados_val = []
mejor_modelo = None
mejor_score = float("inf")
mejor_config = None

# =========================
# 5. ENTRENAR Y VALIDAR
# =========================
for i, cfg in enumerate(configs, start=1):
    print(f"\nEntrenando configuración {i}: {cfg}")

    modelo = HistGradientBoostingRegressor(
        learning_rate=cfg["learning_rate"],
        max_iter=cfg["max_iter"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        l2_regularization=cfg["l2_regularization"],
        random_state=42
    )

    modelo.fit(X_train, y_train)
    pred_val = modelo.predict(X_val)

    res_val = evaluar(y_val, pred_val, f"HGBR_cfg_{i}_val")
    resultados_val.append({
        "config_id": i,
        **cfg,
        "Val_MAE": res_val["MAE"],
        "Val_RMSE": res_val["RMSE"],
        "Val_wMAPE": res_val["wMAPE"]
    })

    print("Resultado validación:", res_val)

    if res_val["MAE"] < mejor_score:
        mejor_score = res_val["MAE"]
        mejor_modelo = modelo
        mejor_config = cfg

tabla_val = pd.DataFrame(resultados_val).sort_values("Val_MAE")

print("\nTabla de resultados en validación:")
print(tabla_val)

print("\nMejor configuración:")
print(mejor_config)

# =========================
# 6. EVALUAR MEJOR MODELO EN TEST
# =========================
pred_test = mejor_modelo.predict(X_test)

res_test = evaluar(y_test, pred_test, "HGBR_best_test")

print("\nResultado final en test:")
print(res_test)