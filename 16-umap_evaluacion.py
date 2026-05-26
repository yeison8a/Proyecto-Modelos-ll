import pandas as pd
import numpy as np
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

# =========================
# CARGAR BASE
# =========================

df = pd.read_csv(
    "base_modelo_horaria.csv"
)

target="demanda_horaria"

features=[
c for c in df.columns
if c not in ["fecha_hora",target]
]

# =========================
# SPLIT TEMPORAL
# =========================

n=len(df)

train_end=int(n*0.7)
val_end=int(n*0.85)

train=df.iloc[:train_end]
val=df.iloc[train_end:val_end]
test=df.iloc[val_end:]

X_train=train[features]
X_val=val[features]
X_test=test[features]

y_train=train[target]
y_val=val[target]
y_test=test[target]

# =========================
# NORMALIZAR
# =========================

scaler=StandardScaler()

X_train=scaler.fit_transform(
X_train
)

X_val=scaler.transform(
X_val
)

X_test=scaler.transform(
X_test
)

# =========================
# UMAP
# =========================

reducer=umap.UMAP(
    n_components=20,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

X_train_umap=reducer.fit_transform(
X_train
)

X_val_umap=reducer.transform(
X_val
)

X_test_umap=reducer.transform(
X_test
)

print(
"\nUMAP componentes:",
X_train_umap.shape[1]
)

# =========================
# MÉTRICAS
# =========================

def evaluar(y,p):

    mae=mean_absolute_error(y,p)

    rmse=np.sqrt(
        mean_squared_error(y,p)
    )

    wmape=(
        np.sum(
            np.abs(y-p)
        )
        /
        np.sum(
            np.abs(y)
        )
    )*100

    return mae,rmse,wmape

# =========================
# HGBR + UMAP
# =========================

print("\nEntrenando HGBR + UMAP")

hgbr=HistGradientBoostingRegressor(
    learning_rate=0.03,
    max_iter=800,
    max_depth=10,
    min_samples_leaf=30,
    l2_regularization=0.1,
    random_state=42
)

hgbr.fit(
X_train_umap,
y_train
)

pred_hgbr=hgbr.predict(
X_test_umap
)

mae,rmse,wmape=evaluar(
y_test,
pred_hgbr
)

print(
"\nHGBR UMAP"
)

print(
f"MAE={mae:.4f}"
)

print(
f"RMSE={rmse:.4f}"
)

print(
f"wMAPE={wmape:.4f}"
)

# =========================
# XGB + UMAP
# =========================

print("\nEntrenando XGB + UMAP")

xgb=XGBRegressor(
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

xgb.fit(
X_train_umap,
y_train
)

pred_xgb=xgb.predict(
X_test_umap
)

mae,rmse,wmape=evaluar(
y_test,
pred_xgb
)

print(
"\nXGB UMAP"
)

print(
f"MAE={mae:.4f}"
)

print(
f"RMSE={rmse:.4f}"
)

print(
f"wMAPE={wmape:.4f}"
)