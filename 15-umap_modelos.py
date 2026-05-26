import umap
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================
# 1. CARGAR BASE
# =========================
df = pd.read_csv(
    "base_modelo_horaria.csv"
)

# variable objetivo
target = "demanda_horaria"

# seleccionar variables predictoras
features = [
    c for c in df.columns
    if c not in ["fecha_hora", target]
]

X = df[features]
y = df[target]

# =========================
# 2. ESCALADO DE VARIABLES
# =========================
# UMAP es sensible a las escalas,
# por eso se normalizan los datos.

scaler = StandardScaler()

X = scaler.fit_transform(X)

# =========================
# 3. REDUCCIÓN DE DIMENSIÓN
# =========================
# crear modelo UMAP
# n_components = número de componentes finales
# n_neighbors = tamaño del vecindario local

reducer = umap.UMAP(
    n_components=20,
    n_neighbors=15,
    random_state=42
)

# ajustar y transformar los datos
X_umap = reducer.fit_transform(X)

# =========================
# 4. RESULTADOS
# =========================
print(
    "Shape UMAP:",
    X_umap.shape
)