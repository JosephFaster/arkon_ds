import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 🔹 Cargar dataset limpio
def load_clean_data(file_name):
    """Carga el dataset en formato Feather."""
    file_path = Path("data") / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo {file_name}")
    return pd.read_feather(file_path)

# 🔹 Feature Engineering
def feature_engineering(df, is_test=False):
    """Genera características adicionales para el modelo."""

    # Extraer día de la semana y hora del viaje
    df["day_of_week"] = df["start_time"].dt.dayofweek  # Lunes=0, Domingo=6
    df["hour"] = df["start_time"].dt.hour  # Hora del día

    # Marcar si es fin de semana
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Crear categorías de duración de viaje
    df["duration_category"] = pd.cut(df["duration"],
                                     bins=[0, 5, 15, 30, 60, 120, 360],
                                     labels=["0-5 min", "5-15 min", "15-30 min", "30-60 min", "60-120 min", "120+ min"])

    # Eliminar `bike_id` (No es relevante para la predicción)
    df.drop(columns=["bike_id"], inplace=True)

    # Convertir variables categóricas a numéricas
    categorical_cols = ["trip_route_category", "start_station", "end_station", "duration_category"]

    # Si es el dataset de entrenamiento, también transformar la variable objetivo
    if not is_test:
        categorical_cols.append("passholder_type")

    # Convertir categorías en números
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    return df

# 🔹 Procesar dataset de entrenamiento
df_clean = load_clean_data("df_clean.feather")
df_features = feature_engineering(df_clean)

# Separar variables predictoras (X) y variable objetivo (y)
X = df_features.drop(columns=["passholder_type"])
y = df_features["passholder_type"]

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Guardar datasets procesados
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_feather("data/train.feather")
test_data.to_feather("data/test.feather")

print("✅ Feature engineering completado. Archivos guardados: train.feather y test.feather")

# 🔹 Procesar dataset de prueba (`df_clean_test.feather`)
df_test = load_clean_data("df_clean_test.feather")
df_test_features = feature_engineering(df_test, is_test=True)

# Guardar dataset de prueba
df_test_features.to_feather("data/test_processed.feather")
print("✅ Dataset de prueba procesado. Archivo guardado: test_processed.feather")
