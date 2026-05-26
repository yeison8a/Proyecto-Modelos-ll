import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# =========================
# CARGAR BASE
# =========================
df = pd.read_csv("base_modelo_horaria.csv")

target = "demanda_horaria"

# seleccionar variables predictoras
features = [
    c for c in df.columns
    if c not in ["fecha_hora", target]
]

# =========================
# TRAIN / VALIDATION / TEST
# =========================
n = len(df)

train_end = int(n*0.70)
val_end = int(n*0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

X_train = train[features]
y_train = train[target]

X_val = val[features]
y_val = val[target]

# =========================
# ESCALAMIENTO
# =========================
# SVR requiere variables escaladas

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# =========================
# CONFIGURACIONES A EVALUAR
# =========================
configs = [

    {"C":1},

    {"C":10},

    {"C":50}

]

# =========================
# ENTRENAMIENTO Y EVALUACIÓN
# =========================
for cfg in configs:

    print("\nEntrenando", cfg)

    # crear modelo SVR
    model = SVR(

        kernel="rbf",

        gamma="scale",

        C=cfg["C"]

    )

    # entrenar modelo
    model.fit(X_train, y_train)

    # predicciones en validación
    pred = model.predict(X_val)

    # métricas
    mae = mean_absolute_error(y_val, pred)

    rmse = np.sqrt(
        mean_squared_error(y_val, pred)
    )

    wmape = (

        np.sum(np.abs(y_val - pred))

        /

        np.sum(np.abs(y_val))

    ) * 100

    # mostrar resultados
    print({

        "MAE": mae,

        "RMSE": rmse,

        "wMAPE": wmape

    })