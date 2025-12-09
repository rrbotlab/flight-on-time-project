import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, '../data/BrFlights2.csv')
model_path = os.path.join(current_dir, 'flight_model_v2.joblib')

# Parámetros ganadores descubiertos en el Notebook
BEST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 8,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 6.33, # Manejo del desbalance
    'random_state': 42,
    'n_jobs': -1
}
OPTIMAL_THRESHOLD = 0.5314

# ==============================================================================
# 1. CARGA Y LIMPIEZA
# ==============================================================================
print("1. Cargando datos...")
try:
    df = pd.read_csv(data_path, encoding='latin1', low_memory=False)
except FileNotFoundError:
    print("❌ Error: No encuentro el CSV BrFlights2.csv en data/")
    exit()

df = df[df['Situacao.Voo'] == 'Realizado']
cols = ['Companhia.Aerea', 'Aeroporto.Origem', 'Aeroporto.Destino', 'Partida.Prevista', 'Partida.Real']
df = df[cols].dropna()

# Target (> 15 min retraso)
df['Partida.Prevista'] = pd.to_datetime(df['Partida.Prevista'])
df['Partida.Real'] = pd.to_datetime(df['Partida.Real'])
df['delay_minutes'] = (df['Partida.Real'] - df['Partida.Prevista']).dt.total_seconds() / 60
df['target'] = np.where(df['delay_minutes'] > 15, 1, 0)

# ==============================================================================
# 2. FEATURE ENGINEERING (Básico)
# ==============================================================================
print("2. Generando variables temporales...")
# Renombrar para consistencia
df = df.rename(columns={'Companhia.Aerea': 'companhia', 'Aeroporto.Origem': 'origem', 'Aeroporto.Destino': 'destino'})

df['hora_partida'] = df['Partida.Prevista'].dt.hour
df['dia_semana'] = df['Partida.Prevista'].dt.dayofweek
df['mes'] = df['Partida.Prevista'].dt.month
df['dia_mes'] = df['Partida.Prevista'].dt.day

# Variables calculadas (Lógica de negocio)
df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
df['es_hora_pico'] = df['hora_partida'].isin([6,7,8,9,17,18,19,20]).astype(int)
df['temporada_alta'] = df['mes'].isin([1,2,7,12]).astype(int)
df['fin_inicio_mes'] = df['dia_mes'].isin(list(range(1,6)) + list(range(26,32))).astype(int)

# ==============================================================================
# 3. SPLIT Y CORRECCIÓN DE DATA LEAKAGE
# ==============================================================================
print("3. Separando Train/Test ANTES de calcular estadísticas (Anti-Trampa)...")

# Guardamos columnas necesarias para stats temporalmente
cols_for_stats = ['target', 'delay_minutes']
X = df.drop(columns=cols_for_stats) 
y = df['target']

# Mantenemos índices para recuperar delay_minutes después del split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Recuperamos 'target' y 'delay' SOLO para el conjunto de entrenamiento
train_full = X_train_raw.copy()
train_full['target'] = y_train
train_full['delay_minutes'] = df.loc[train_full.index, 'delay_minutes']

# ==============================================================================
# 4. CÁLCULO DE ESTADÍSTICAS (Solo con Train)
# ==============================================================================
print("4. Calculando estadísticas históricas...")
stats_companhia = train_full.groupby('companhia')['delay_minutes'].mean().to_dict()
stats_ruta = train_full.groupby(['origem', 'destino'])['delay_minutes'].mean().to_dict()
stats_hora = train_full.groupby('hora_partida')['target'].mean().to_dict()

# Función para aplicar estadísticas (mapeo seguro)
def apply_stats(dataset):
    ds = dataset.copy()
    ds['airline_avg_delay'] = ds['companhia'].map(stats_companhia).fillna(0)
    ds['route_avg_delay'] = ds.set_index(['origem', 'destino']).index.map(stats_ruta).fillna(0)
    ds['hour_delay_rate'] = ds['hora_partida'].map(stats_hora).fillna(0)
    return ds

# Aplicamos a Train (para entrenar) y a Test (para validar)
X_train = apply_stats(X_train_raw)

# ==============================================================================
# 5. ENCODING
# ==============================================================================
print("5. Codificando categorías...")
le_companhia = LabelEncoder()
le_origem = LabelEncoder()
le_destino = LabelEncoder()

# Ajustamos encoders solo con Train
X_train['companhia_encoded'] = le_companhia.fit_transform(X_train['companhia'].astype(str))
X_train['origem_encoded'] = le_origem.fit_transform(X_train['origem'].astype(str))
X_train['destino_encoded'] = le_destino.fit_transform(X_train['destino'].astype(str))

# ==============================================================================
# 6. ENTRENAMIENTO FINAL (XGBoost)
# ==============================================================================
features_finales = [
    'companhia_encoded', 'origem_encoded', 'destino_encoded',
    'hora_partida', 'dia_semana', 'mes', 
    'es_fin_semana', 'es_hora_pico', 'temporada_alta', 'fin_inicio_mes',
    'airline_avg_delay', 'route_avg_delay', 'hour_delay_rate'
]

print(f"6. Entrenando XGBoost con parámetros optimizados del Notebook...")
model = xgb.XGBClassifier(**BEST_PARAMS)
model.fit(X_train[features_finales], y_train)

# ==============================================================================
# 7. GUARDADO
# ==============================================================================
artifacts = {
    'model': model,
    'features': features_finales,
    'encoders': {
        'companhia': le_companhia,
        'origem': le_origem,
        'destino': le_destino
    },
    'stats': {
        'companhia': stats_companhia,
        'ruta': stats_ruta,
        'hora': stats_hora
    },
    'threshold': OPTIMAL_THRESHOLD 
}

joblib.dump(artifacts, model_path)
print(f"✅ ¡Éxito! Modelo V3 Final guardado en: {model_path}")
print(f"   Threshold óptimo configurado: {OPTIMAL_THRESHOLD}")