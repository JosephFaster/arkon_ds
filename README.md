# arkon_ds
 Prueba técnica para Arkon data
# 🚴‍♂️ Predicción de Tipos de Pase en un Sistema de Bicicletas Compartidas

## 📌 Descripción
Este repositorio contiene el código y los datos utilizados en el análisis y modelado para la predicción del tipo de pase (`passholder_type`) en un sistema de bicicletas compartidas. A lo largo del desarrollo, se realizaron múltiples modificaciones y experimentos con diferentes enfoques de preprocesamiento, feature engineering y modelos de machine learning.

Si se exploran los **commits**, se puede ver que el repositorio fue modificado muchas veces, esto sin contar los archivos que nunca fueron indexados a Git. Hubo varias iteraciones de preprocesamiento, optimización de modelos y cambios en la organización del código.

## 📂 Estructura del Repositorio

```
├── data/                  # Contiene todos los datos utilizados en el análisis y modelado
├── notebooks/             # Incluye los notebooks con el flujo de trabajo y experimentos
├── docker/                # Carpeta vacía, se planeaba contenerización pero no se alcanzó
├── notebooks/             # Notebooks con el análisis y pruebas de modelos
├── main.py                # Script principal (en caso de implementación)
├── label_encoder.pkl      # Modelo de codificación de etiquetas guardado
├── lightgbm_model.pkl     # Modelo LightGBM entrenado
├── requirements.txt       # Dependencias necesarias para reproducir el entorno
├── Dockerfile             # Archivo para contenerización (no implementado por falta de tiempo)
└── README.md              # Documentación del proyecto
```

## ⚙️ Modelos y Resultados
Se probaron varios modelos de Machine Learning, incluyendo **Random Forest, XGBoost y LightGBM**. Los modelos más complejos alcanzaron una precisión del **92% en promedio**, pero tenían un peso de alrededor de **1 GB cada uno**, lo que impidió subirlos a GitHub, incluso usando Git LFS.

Después de varias pruebas de **feature engineering** y optimización, se optó por un modelo más ligero para garantizar que los recursos fueran manejables sin sacrificar demasiada precisión.

## 🚀 Desafíos y Limitaciones
1. **Tiempo y recursos:** Se intentó implementar un sistema más robusto con **Docker**, pero por falta de tiempo no se logró poner en producción.
2. **Procesamiento de datos:** Se realizaron múltiples estrategias de limpieza y preprocesamiento, pero el formato de `submission` en Kaggle limitó algunas decisiones.
3. **Espacio en GitHub:** Debido al tamaño de los modelos, se tuvieron que descartar algunos y trabajar con versiones más livianas.
4. **Pipeline automatizado:** Se generó un flujo de trabajo para facilitar la replicabilidad del modelo, pero algunas partes quedaron en experimentación.

## 📌 Notas Finales
El proyecto pasó por muchas iteraciones y pruebas de modelado antes de llegar a la versión final. La carpeta `docker/` quedó vacía porque no se alcanzó a implementar la puesta en producción, y la estructura de carpetas fue reorganizada varias veces para optimizar el flujo de trabajo.

Este README resume los aspectos más relevantes del repositorio y del proceso de desarrollo. ¡Gracias por revisar el proyecto! 🚴‍♂️📊



