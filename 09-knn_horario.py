import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================
# CARGAR DATOS
# =====================

df = pd.read_csv("base_modelo_horaria.csv")

target = "demanda_horaria"

features = [
    c for c in df.columns
    if c not in ["fecha_hora", target]
]

n=len(df)

train_end=int(n*0.70)
val_end=int(n*0.85)

train=df.iloc[:train_end]
val=df.iloc[train_end:val_end]
test=df.iloc[val_end:]

X_train=train[features]
y_train=train[target]

X_val=val[features]
y_val=val[target]

X_test=test[features]
y_test=test[target]

# =====================
# ESCALADO
# =====================

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)

X_val=scaler.transform(X_val)

X_test=scaler.transform(X_test)

# =====================
# MÉTRICAS
# =====================

def evaluar(y,pred):

    mae=mean_absolute_error(y,pred)

    rmse=np.sqrt(mean_squared_error(y,pred))

    wmape=np.sum(np.abs(y-pred))/np.sum(np.abs(y))*100

    return mae,rmse,wmape

# =====================
# GRID SEARCH SIMPLE
# =====================

configs=[5,10,20,30]

resultados=[]

for k in configs:

    print(f"\nEntrenando KNN k={k}")

    model=KNeighborsRegressor(
        n_neighbors=k,
        weights="distance"
    )

    model.fit(X_train,y_train)

    pred=model.predict(X_val)

    mae,rmse,wmape=evaluar(y_val,pred)

    resultados.append([k,mae,rmse,wmape])

tabla=pd.DataFrame(
    resultados,
    columns=[
        "k",
        "Val_MAE",
        "Val_RMSE",
        "Val_wMAPE"
    ]
)

print(tabla.sort_values("Val_MAE"))