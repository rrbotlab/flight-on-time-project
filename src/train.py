import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 1. Configurar rutas (Para que funcione en Mac/Windows)
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, '../data/BrFlights2.csv')
model_path = os.path.join(current_dir, 'flight_model_v1.joblib')

print(f"Leyendo datos desde: {data_path}")

# 2. Cargar y Limpiar
try:
    df = pd.read_csv(data_path, encoding='latin1', low_memory=False)
except FileNotFoundError:
    print("ERROR: No encuentro el CSV. Asegúrate de que 'BrFlights2.csv' esté en la carpeta 'data'.")
    exit()

df = df[df['Situacao.Voo'] == 'Realizado']
df = df[['Companhia.Aerea', 'Aeroporto.Origem', 'Aeroporto.Destino', 'Partida.Prevista', 'Partida.Real']].dropna()

# Crear Target
df['Partida.Prevista'] = pd.to_datetime(df['Partida.Prevista'])
df['Partida.Real'] = pd.to_datetime(df['Partida.Real'])
df['delay_minutes'] = (df['Partida.Real'] - df['Partida.Prevista']).dt.total_seconds() / 60
df['target'] = np.where(df['delay_minutes'] > 15, 1, 0)

# Features básicas
df['hora_partida'] = df['Partida.Prevista'].dt.hour
df['dia_semana'] = df['Partida.Prevista'].dt.dayofweek
df['mes'] = df['Partida.Prevista'].dt.month

# Renombrar
df = df.rename(columns={
    'Companhia.Aerea': 'companhia',
    'Aeroporto.Origem': 'origem',
    'Aeroporto.Destino': 'destino'
})

# 3. Encoding
le_companhia = LabelEncoder()
le_origem = LabelEncoder()
le_destino = LabelEncoder()

df['companhia_encoded'] = le_companhia.fit_transform(df['companhia'].astype(str))
df['origem_encoded'] = le_origem.fit_transform(df['origem'].astype(str))
df['destino_encoded'] = le_destino.fit_transform(df['destino'].astype(str))

# 4. Entrenar Modelo Rápido
X = df[['companhia_encoded', 'origem_encoded', 'destino_encoded', 'hora_partida', 'dia_semana', 'mes']]
y = df['target']

print("Entrenando modelo...")
model = RandomForestClassifier(n_estimators=10, max_depth=10, class_weight='balanced', random_state=42)
model.fit(X, y)

# 5. Guardar
artifacts = {
    'model': model,
    'le_companhia': le_companhia,
    'le_origem': le_origem,
    'le_destino': le_destino
}

joblib.dump(artifacts, model_path)
print(f"✅ Modelo guardado exitosamente en: {model_path}")