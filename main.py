from fastapi import FastAPI
import pickle
import pandas as pd

# Cargar el modelo entrenado
with open("lightgbm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Iniciar FastAPI
app = FastAPI()

# Endpoint para verificar si la API está corriendo
@app.get("/")
def read_root():
    return {"message": "API para predicciones de Passholder Type está activa 🚀"}

# Endpoint de predicción
@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}