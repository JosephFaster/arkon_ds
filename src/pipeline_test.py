import pandas as pd
import numpy as np
import re
from pathlib import Path

def find_file(filename, search_dir=None):
    """
    Busca un archivo en la carpeta 'data' o en la jerarquía de carpetas.
    """
    search_dir = Path(search_dir) if search_dir else Path.cwd()

    # Buscar en la carpeta "data"
    file_path = search_dir / "data" / filename
    if file_path.exists():
        return str(file_path)

    # Búsqueda en la jerarquía de directorios
    for parent in search_dir.parents:
        possible_path = parent / "data" / filename
        if possible_path.exists():
            return str(possible_path)

    raise FileNotFoundError(f'No se encontró el archivo {filename} en la carpeta "data".')

def normalize_date_format(date_str):
    """Convierte fechas a formato estándar YYYY-MM-DD HH:MM:SS."""
    if pd.isna(date_str):
        return np.nan

    if re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', date_str):
        return date_str

    if re.match(r'\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}', date_str):
        return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M').strftime('%Y-%m-%d %H:%M:%S')

    if re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
        return pd.to_datetime(date_str, format='%m/%d/%Y').strftime('%Y-%m-%d 00:00:00')

    return np.nan

def clean_data(df, is_test=False):
    """Pipeline de limpieza de datos, ajustado para sets de entrenamiento y prueba."""

    # Normalizar fechas
    df['start_time'] = df['start_time'].astype(str).apply(normalize_date_format)
    df['end_time'] = df['end_time'].astype(str).apply(normalize_date_format)

    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

    # Convertir tipos de datos
    if not is_test and 'passholder_type' in df.columns:  # Solo si no es dataset de prueba y la columna existe
        df['passholder_type'] = df['passholder_type'].astype('category')

    df['trip_route_category'] = df['trip_route_category'].astype('category')
    df['bike_id'] = df['bike_id'].astype(str)  # Mantener como string
    df['start_station'] = df['start_station'].astype('category')
    df['end_station'] = df['end_station'].astype('category')

    # Convertir plan_duration a entero si la columna existe en el dataset
    if 'plan_duration' in df.columns:
        df['plan_duration'] = df['plan_duration'].fillna(0).astype(int)

    # Eliminar valores nulos
    df.dropna(inplace=True)

    # Filtrar valores atípicos en duración del viaje (máximo 6 horas)
    df = df[df['duration'] <= 360]

    # Filtrar coordenadas fuera de Los Ángeles
    lat_min, lat_max = 33.7, 34.3
    lon_min, lon_max = -118.5, -118.15
    mask = (
        (df['start_lat'].between(lat_min, lat_max)) & 
        (df['start_lon'].between(lon_min, lon_max)) & 
        (df['end_lat'].between(lat_min, lat_max)) & 
        (df['end_lon'].between(lon_min, lon_max))
    )

    df = df[mask].copy()

    return df

# Buscar la ruta del archivo test_set.csv
try:
    test_set_path = find_file("test_set.csv")
    print(f"✅ Archivo encontrado en: {test_set_path}")
except FileNotFoundError as e:
    print(e)
    exit(1)

# Cargar el archivo test_set.csv con dtype correcto para evitar warning
dtype_spec = {
    'bike_id': str  # Forzar que bike_id sea string desde el inicio
}

test_data = pd.read_csv(test_set_path, dtype=dtype_spec, low_memory=False)

# Limpieza de datos (es dataset de prueba, por lo que no tiene `passholder_type` ni `plan_duration`)
df_clean_test = clean_data(test_data, is_test=True)

# Guardar el dataset limpio en formato Feather
output_path = Path("data/df_clean_test.feather")
df_clean_test.to_feather(output_path)

print(f"✅ Dataset de test procesado y guardado en: {output_path}")
