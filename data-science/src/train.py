# data-science/src/train.py
import pandas as pd
import numpy as np
import joblib
import holidays
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

# --- CONFIGURAÇÃO ---
print(" Iniciando Treinamento V4.1 (Native Categorical)...")
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, '../data/raw/BrFlights_Enriched_v4.csv')
model_path = os.path.join(current_dir, 'flight_classifier_v4.joblib')

# 1. CARGA
if not os.path.exists(data_path):
    print(f" Erro: Dataset não encontrado em {data_path}")
    exit()
    
df = pd.read_csv(data_path, low_memory=False)

# 2. PREPARAÇÃO E LIMPEZA
print("🛠️  Preparando dados...")

# Converter colunas numéricas
for col in ['LatOrig', 'LongOrig', 'LatDest', 'LongDest', 'precipitation', 'wind_speed']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Haversine Distance
def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

df['distancia_km'] = haversine(df['LatOrig'], df['LongOrig'], df['LatDest'], df['LongDest'])

# Datas
df['Partida.Prevista'] = pd.to_datetime(df['Partida.Prevista'])
df['Partida.Real'] = pd.to_datetime(df['Partida.Real'])
df = df.dropna(subset=['Partida.Real', 'Partida.Prevista'])

# Target (> 15 min atraso)
delay = (df['Partida.Real'] - df['Partida.Prevista']).dt.total_seconds() / 60
df['target'] = np.where(delay > 15, 1, 0)

# Features
br_holidays = holidays.Brazil()
df['is_holiday'] = df['Partida.Prevista'].dt.date.apply(lambda x: 1 if x in br_holidays else 0)
df['hora'] = df['Partida.Prevista'].dt.hour
df['dia_semana'] = df['Partida.Prevista'].dt.dayofweek
df['mes'] = df['Partida.Prevista'].dt.month
df['clima_imputado'] = df.get('clima_imputado', 0)

# Renomear
df.rename(columns={'Companhia.Aerea': 'companhia', 'Aeroporto.Origem': 'origem', 'Aeroporto.Destino': 'destino'}, inplace=True)

# Seleção Final
features = ['companhia', 'origem', 'destino', 'distancia_km', 'hora', 'dia_semana', 'mes', 'is_holiday', 'precipitation', 'wind_speed', 'clima_imputado']
cat_features = ['companhia', 'origem', 'destino']

# Garantir strings para o CatBoost
for col in cat_features:
    df[col] = df[col].astype(str)

X = df[features]
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. TREINAMENTO (NATIVE)
print(" Treinando CatBoost (Native Mode)...")
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    auto_class_weights='Balanced',
    cat_features=cat_features,
    verbose=100,
    allow_writing_files=False,
    random_seed=42
)

model.fit(X_train, y_train)

# 4. VALIDAÇÃO (THRESHOLD 0.35)
THRESHOLD = 0.35
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= THRESHOLD).astype(int)
rec = recall_score(y_test, preds)

print("-" * 30)
print(f"🏆 RECALL VALIDADO (Corte {THRESHOLD}): {rec:.2%}")
print("-" * 30)

# 5. EXPORTAÇÃO (FULL)
print(" Gerando modelo final...")
model_final = CatBoostClassifier(
    iterations=500, learning_rate=0.1, depth=6, auto_class_weights='Balanced',
    cat_features=cat_features, verbose=False, allow_writing_files=False
)
model_final.fit(X, y) # Treina com tudo

artifact = {
    'model': model_final,
    'features': features,
    'cat_features': cat_features,
    'metadata': {'versao': '4.1-Native', 'threshold': THRESHOLD, 'recall': rec}
}

joblib.dump(artifact, model_path)
print(f"✅ Salvo em: {model_path}")