import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import os

os.makedirs("resultados_dim",exist_ok=True)

# ======================
# CARGAR BASE
# ======================

df = pd.read_csv("base_modelo_horaria.csv")

target = "demanda_horaria"

features = [c for c in df.columns
            if c not in ["fecha_hora",target]]

X = df[features]
y = df[target]

# ======================
# CORRELACIÓN
# ======================

corr = df[features+[target]].corr()[target]

corr = corr.drop(target)

corr_abs = corr.abs().sort_values(
    ascending=False
)

corr_abs.to_csv(
    "resultados_dim/correlaciones.csv"
)

print(corr_abs.head(20))

# top20 correlaciones

plt.figure(figsize=(10,8))

corr_abs.head(20).sort_values().plot(
    kind="barh"
)

plt.xlabel("Correlación absoluta")

plt.title(
"Top 20 variables más correlacionadas"
)

plt.tight_layout()

plt.savefig(
"resultados_dim/top20_correlacion.png",
dpi=200
)

plt.show()

# ======================
# IMPORTANCIA XGB
# ======================

model = XGBRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X,y)

imp = pd.Series(
    model.feature_importances_,
    index=features
)

imp = imp.sort_values(
    ascending=False
)

imp.to_csv(
"resultados_dim/importancias_xgb.csv"
)

plt.figure(figsize=(10,8))

imp.head(20).sort_values().plot(
    kind="barh"
)

plt.xlabel("Importancia")

plt.title(
"Top 20 variables importantes"
)

plt.tight_layout()

plt.savefig(
"resultados_dim/top20_importancia.png",
dpi=200
)

plt.show()

# ======================
# VARIABLES CANDIDATAS
# ======================

bajas = corr_abs[corr_abs<0.02]

print("\nVariables candidatas:")

print(bajas.index.tolist())