import pandas as pd

# 1. cargar base final horaria
df = pd.read_csv("base_modelo_horaria.csv")
df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

# 2. definir variable objetivo
target = "demanda_horaria"

# 3. definir variables de entrada
features = [c for c in df.columns if c not in ["fecha_hora", target]]

# 4. split temporal
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

# 5. separar X e y
X_train = train[features]
y_train = train[target]

X_val = val[features]
y_val = val[target]

X_test = test[features]
y_test = test[target]

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

print("\nNúmero de variables:", len(features))

print("\nRangos de fechas:")
print("Train:", train["fecha_hora"].min(), "->", train["fecha_hora"].max())
print("Val:", val["fecha_hora"].min(), "->", val["fecha_hora"].max())
print("Test:", test["fecha_hora"].min(), "->", test["fecha_hora"].max())