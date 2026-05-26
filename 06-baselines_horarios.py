import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. cargar base final horaria
df = pd.read_csv("base_modelo_horaria.csv")
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

target = "demanda_horaria"
features = [c for c in df.columns if c not in ["fecha_hora", target]]

# 2. split temporal
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

X_val = val[features]
y_val = val[target]

X_test = test[features]
y_test = test[target]

# 3. función de evaluación
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

# 4. baselines
baselines = {
    "Baseline_lag_1": "demanda_horaria_lag_1",
    "Baseline_lag_24": "demanda_horaria_lag_24",
    "Baseline_lag_168": "demanda_horaria_lag_168"
}

resultados = []

for nombre, col in baselines.items():
    pred_val = X_val[col].values
    res_val = evaluar(y_val, pred_val, nombre + "_val")

    pred_test = X_test[col].values
    res_test = evaluar(y_test, pred_test, nombre + "_test")

    resultados.append({
        "Modelo": nombre,
        "Val_MAE": res_val["MAE"],
        "Val_RMSE": res_val["RMSE"],
        "Val_wMAPE": res_val["wMAPE"],
        "Test_MAE": res_test["MAE"],
        "Test_RMSE": res_test["RMSE"],
        "Test_wMAPE": res_test["wMAPE"]
    })

tabla_resultados = pd.DataFrame(resultados).sort_values("Test_MAE")
print(tabla_resultados)