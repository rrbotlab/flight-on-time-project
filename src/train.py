import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# --- CONFIGURACIÓN ---
print("🚀 Iniciando entrenamiento del modelo MVP (Random Forest)...")
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, '../data/BrFlights2.csv')
model_path = os.path.join(current_dir, 'flight_model_mvp.joblib')

# 1. CARGA DE DATOS
try:
    df = pd.read_csv(data_path, encoding='latin1', low_memory=False)
except FileNotFoundError:
    print("❌ Error: No se encuentra BrFlights2.csv en la carpeta data/")
    exit()

# Filtros básicos
df = df[df['Situacao.Voo'] == 'Realizado']
cols = ['Companhia.Aerea', 'Aeroporto.Origem', 'Aeroporto.Destino', 'Partida.Prevista', 'Partida.Real']
df = df[cols].dropna()

# 2. FEATURE ENGINEERING (Igual al Notebook Untitled35)
print("⚙️ Generando features...")
df['Partida.Prevista'] = pd.to_datetime(df['Partida.Prevista'])
df['Partida.Real'] = pd.to_datetime(df['Partida.Real'])

# Target: Atraso > 15 minutos
df['delay_minutes'] = (df['Partida.Real'] - df['Partida.Prevista']).dt.total_seconds() / 60
df['target'] = np.where(df['delay_minutes'] > 15, 1, 0)

# Variables temporales
df['hora'] = df['Partida.Prevista'].dt.hour
df['dia_semana'] = df['Partida.Prevista'].dt.dayofweek
df['mes'] = df['Partida.Prevista'].dt.month

# Renombrar columnas
df = df.rename(columns={
    'Companhia.Aerea': 'companhia',
    'Aeroporto.Origem': 'origem',
    'Aeroporto.Destino': 'destino'
})

# 3. ENCODING
print("🔠 Codificando categorías...")
encoders = {}
cat_features = ['companhia', 'origem', 'destino']

for col in cat_features:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[f'{col}_encoded'] = le.fit_transform(df[col])
    encoders[col] = le  # Guardamos el encoder para la API

# 4. ENTRENAMIENTO
features_finais = ['companhia_encoded', 'origem_encoded', 'destino_encoded', 'hora', 'dia_semana', 'mes']
X = df[features_finais]
y = df['target']

print("🧠 Entrenando Random Forest (Esto puede tardar unos segundos)...")
# Parámetros del Notebook Untitled35
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

# 5. GUARDAR ARTEFACTOS
print("💾 Guardando modelo...")
production_artifact = {
    'model': model,
    'encoders': encoders,
    'features': features_finais,
    'metadata': {
        'version': '1.0 MVP',
        'author': 'FlightOnTime DS Team',
        'description': 'Random Forest MVP'
    }
}

joblib.dump(production_artifact, model_path)
print(f"✅ ¡Éxito! Modelo guardado en: {model_path}")