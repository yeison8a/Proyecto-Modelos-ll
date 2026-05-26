import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

# ==========================
# CARGAR BASE
# ==========================

df = pd.read_csv("base_modelo_horaria.csv")

target = "demanda_horaria"

features = [c for c in df.columns
            if c not in ["fecha_hora",target]]

X = df[features]
y = df[target]

# ==========================
# SPLIT TEMPORAL
# ==========================

n=len(df)

train_end=int(n*0.70)
val_end=int(n*0.85)

X_train=X.iloc[:train_end]
y_train=y.iloc[:train_end]

X_val=X.iloc[train_end:val_end]
y_val=y.iloc[train_end:val_end]

X_test=X.iloc[val_end:]
y_test=y.iloc[val_end:]

# ==========================
# ESCALADO
# ==========================

scaler=StandardScaler()

X_train_sc=scaler.fit_transform(X_train)

X_val_sc=scaler.transform(X_val)

X_test_sc=scaler.transform(X_test)

# ==========================
# PCA
# ==========================

pca=PCA(n_components=0.95)

X_train_pca=pca.fit_transform(X_train_sc)

X_val_pca=pca.transform(X_val_sc)

X_test_pca=pca.transform(X_test_sc)

print("Variables originales:",X.shape[1])
print("Componentes PCA:",X_train_pca.shape[1])

# ==========================
# MÉTRICAS
# ==========================

def evaluar(y_true,y_pred):

    mae=mean_absolute_error(y_true,y_pred)

    rmse=np.sqrt(
        mean_squared_error(y_true,y_pred)
    )

    wmape=(
        np.sum(np.abs(y_true-y_pred))
        /
        np.sum(np.abs(y_true))
    )*100

    return mae,rmse,wmape

# ==========================
# HGBR + PCA
# ==========================

print("\nEntrenando HGBR + PCA")

hgbr=HistGradientBoostingRegressor(

    learning_rate=0.03,
    max_iter=800,
    max_depth=10,
    min_samples_leaf=30,
    l2_regularization=0.1,
    random_state=42
)

hgbr.fit(X_train_pca,y_train)

pred_h=hgbr.predict(X_test_pca)

mae_h,rmse_h,wmape_h=evaluar(
    y_test,
    pred_h
)

print("\nHGBR PCA")

print(f"MAE={mae_h:.4f}")
print(f"RMSE={rmse_h:.4f}")
print(f"wMAPE={wmape_h:.4f}")

# ==========================
# XGB + PCA
# ==========================

print("\nEntrenando XGB + PCA")

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

xgb.fit(X_train_pca,y_train)

pred_x=xgb.predict(X_test_pca)

mae_x,rmse_x,wmape_x=evaluar(
    y_test,
    pred_x
)

print("\nXGB PCA")

print(f"MAE={mae_x:.4f}")
print(f"RMSE={rmse_x:.4f}")
print(f"wMAPE={wmape_x:.4f}")