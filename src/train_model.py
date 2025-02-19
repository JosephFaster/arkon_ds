import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# 🔹 Cargar los datos de entrenamiento y prueba
def load_data(file_name):
    """Carga el dataset en formato Feather."""
    file_path = Path("data") / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo {file_name}")
    return pd.read_feather(file_path)

print("Cargando datasets...")
train_data = load_data("train.feather")
test_data = load_data("test.feather")

# 🔹 Separar variables predictoras (X) y variable objetivo (y)
X_train = train_data.drop(columns=["passholder_type"])
y_train = train_data["passholder_type"]

X_test = test_data.drop(columns=["passholder_type"])
y_test = test_data["passholder_type"]

# 🔹 Normalizar datos (si es necesario)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el scaler
joblib.dump(scaler, "models/scaler.pkl")

# 🔹 Definir modelos a entrenar
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=500)
}

# 🔹 Entrenar y evaluar modelos
best_model = None
best_score = 0

for model_name, model in models.items():
    print(f"🔍 Entrenando {model_name}...")
    model.fit(X_train_scaled, y_train)
    
    # Validación cruzada
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
    mean_score = np.mean(scores)
    
    print(f"✅ {model_name} - Accuracy promedio: {mean_score:.4f}")

    # Evaluación en test set
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"📊 {model_name} - Accuracy en test: {test_accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Guardar el mejor modelo
    if test_accuracy > best_score:
        best_score = test_accuracy
        best_model = model
        best_model_name = model_name

# 🔹 Guardar el mejor modelo
if best_model:
    model_path = Path("models") / f"{best_model_name}_best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"🎯 Mejor modelo guardado: {best_model_name} en {model_path}")

print("✅ Entrenamiento completado.")
