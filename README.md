# Predicción de Demanda de Servicios de Ambulancia para la Planeación Operativa

Proyecto de **Machine Learning** enfocado en la predicción de demanda horaria de servicios médicos de emergencia utilizando el dataset **EMS Incident Dispatch Data (NYC Open Data)**.

El proyecto incluye:

- Construcción de base temporal horaria.
- Ingeniería de características.
- Entrenamiento y comparación de modelos de ML.
- Ensamble de modelos.
- Reducción de dimensión con **PCA** y **UMAP**.
- Generación de gráficas y resultados del informe.

---

# Requisitos

Instalar las dependencias necesarias:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost umap-learn
```

---

# Dataset

El archivo original **NO está incluido en el repositorio** debido a su gran tamaño (>5 GB).

Debe descargarse manualmente desde:

**NYC Open Data — EMS Incident Dispatch Data**

https://data.cityofnewyork.us/Public-Safety/EMS-Incident-Dispatch-Data/76xm-jjuj

Después de descargarlo, colocar el archivo:

```text
ems-incident-dispatch-data.csv
```

en la carpeta raíz del proyecto.

---

# Flujo de ejecución

Ejecutar los scripts en el siguiente orden:

### 1. Construcción de base horaria

```bash
python 01-base_horaria.py
```

Genera la demanda agregada por hora.

---

### 2. Exploración de categorías

```bash
python 02-scan_categorias.py
```

Explora variables categóricas del dataset.

---

### 3. Enriquecimiento de la base

```bash
python 03-enriquecer_base_horaria.py
```

Añade variables derivadas del dataset EMS.

---

### 4. Ingeniería de características

```bash
python 04-crear_features_horarias.py
```

Genera lags, medias móviles y features temporales.

---

### 5. División temporal

```bash
python 05-split_horario.py
```

Crea conjuntos train, validation y test.

---

### 6. Baselines

```bash
python 06-baselines_horarios.py
```

Entrena modelos de referencia.

---

### 7. HistGradientBoosting

```bash
python 07-hgbr_horario.py
```

Entrena modelo HGBR.

---

### 8. XGBoost

```bash
python 08-xgboost_horario.py
```

Entrena modelo XGBoost.

---

### 9. KNN

```bash
python 09-knn_horario.py
```

Evalúa configuraciones KNN.

---

### 10. SVR

```bash
python 10-svr_horario.py
```

Entrena modelo Support Vector Regression.

---

### 11. Ensamble final

```bash
python 11-xgb_poisson_peak_ensamble.py
```

Combina modelos HGBR y XGBoost.

---

### 12. Análisis de picos

```bash
python 12-analizar_picos_completo.py
```

Analiza errores en horas de alta demanda.

---

### 13. Análisis de variables

```bash
python 13-analisis_variables.py
```

Calcula correlaciones y variables candidatas a eliminación.

---

### 14. PCA

```bash
python 14-pca_modelos.py
```

Evalúa reducción lineal de dimensión.

---

### 15. UMAP

```bash
python 15-umap_modelos.py
```

Genera representación reducida no lineal.

---

### 16. Evaluación UMAP

```bash
python 16-umap_evaluacion.py
```

Evalúa modelos usando componentes UMAP.

---

### 17. Gráficas del documento

```bash
python 17-graficas_documento.py
```

Genera figuras y tablas usadas en el informe.

---

# Resultados

Los archivos generados se almacenan en:

```text
resultados/
resultados_dim/
```

incluyendo:

- tablas de métricas
- predicciones
- gráficas
- resultados de reducción de dimensión