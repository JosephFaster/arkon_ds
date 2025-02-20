from fastapi import FastAPI
import pandas as pd
import pickle
import os
from pydantic import BaseModel
from typing import List

# Inicializar FastAPI
app = FastAPI(title="API de Predicciones con LightGBM")

# Verificar que los archivos existen antes de cargarlos
model_path = "lightgbm_model.pkl"
encoder_path = "label_encoder.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"ERROR: El archivo {model_path} no fue encontrado.")

if not os.path.exists(encoder_path):
    raise FileNotFoundError(f"ERROR: El archivo {encoder_path} no fue encontrado.")

# Cargar el modelo LightGBM
with open(model_path, "rb") as f:
    lgb_model = pickle.load(f)

# Cargar el LabelEncoder
with open(encoder_path, "rb") as f:
    label_encoder = pickle.load(f)


# Definir el esquema de entrada con Pydantic
class PredictionInput(BaseModel):
    trip_id: int
    duration: int
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    trip_route_category: str
    start_station: str
    end_station: str
    hour: int
    day_of_week: str
    year_month: str
    year: int

class PredictionResponse(BaseModel):
    trip_id: int
    passholder_type: str

@app.get("/")
def home():
    return {"message": "API funcionando correctamente"}

@app.post("/predict", response_model=List[PredictionResponse])
def predict(data: List[PredictionInput]):
    df_test = pd.DataFrame([d.dict() for d in data])

    # Asegurar que las variables categóricas tengan las mismas categorías que en entrenamiento
    categorical_columns = ["trip_route_category", "start_station", "end_station", "day_of_week", "year_month"]
    for col in categorical_columns:
        df_test[col] = df_test[col].astype("category")
        df_test[col] = df_test[col].cat.set_categories(label_encoder.classes_, ordered=False)

    # Asegurar que el df_test solo tenga las columnas esperadas
    feature_columns = [
        "duration", "start_lat", "start_lon", "end_lat", "end_lon",
        "trip_route_category", "start_station", "end_station", "hour",
        "day_of_week", "year_month", "year"
    ]
    X_test = df_test[feature_columns]

    # Hacer la predicción
    y_test_pred_proba = lgb_model.predict(X_test)
    y_test_pred = y_test_pred_proba.argmax(axis=1)
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)

    # Crear la respuesta
    results = [
        {"trip_id": row["trip_id"], "passholder_type": pred}
        for row, pred in zip(data, y_test_pred_labels)
    ]

    return results
