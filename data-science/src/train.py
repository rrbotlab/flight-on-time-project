import pandas as pd
import numpy as np
import joblib
import holidays
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. DEFINIÇÃO DA CLASSE SAFE ENCODER (CRÍTICO) ---
# Esta classe precisa estar aqui para ser salva junto com o modelo
class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.classes_ = {}
        self.unknown_token = -1  # Valor para categorias desconhecidas

    def fit(self, y):
        # Aprende apenas as classes presentes no fit (Treino)
        unique_labels = pd.Series(y).unique()
        self.classes_ = {label: idx for idx, label in enumerate(unique_labels)}
        return self

    def transform(self, y):
        # Transforma, mapeando desconhecidos para -1 (sem erro)
        return pd.Series(y).apply(lambda x: self.classes_.get(x, self.unknown_token))

# --- FUNÇÕES AUXILIARES ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula distância em KM entre coordenadas"""
    r = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# --- CONFIGURAÇÃO ---
print(" Iniciando treinamento V3.0-CAT (CatBoost + SafeEncoder)...")
current_dir = os.path.dirname(__file__)
# Ajuste este caminho conforme sua estrutura de pastas
data_path = os.path.join(current_dir, '../data/BrFlights2.csv') 
model_path = os.path.join(current_dir, 'flight_classifier_mvp.joblib')

# 2. CARGA DE DADOS
try:
    df = pd.read_csv(data_path, encoding='latin1', low_memory=False)
    print(f" Registros carregados: {len(df)}")
except FileNotFoundError:
    print(f" Erro: Arquivo não encontrado em {data_path}")
    exit()

# 3. LIMPEZA E ENGENHARIA (CORRIGIDO)
print(" Criando features e aplicando limpeza...")

# A. Distância
df['distancia_km'] = haversine_distance(
    pd.to_numeric(df['LatOrig'], errors='coerce'), 
    pd.to_numeric(df['LongOrig'], errors='coerce'), 
    pd.to_numeric(df['LatDest'], errors='coerce'), 
    pd.to_numeric(df['LongDest'], errors='coerce')
)

# B. Datas
cols_datas = ['Partida.Prevista', 'Partida.Real', 'Chegada.Real']
for col in cols_datas:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# C. Filtragem Básica (Sem erro de índice)
# Primeiro filtramos, depois removemos nulos
df_clean = df[df['Situacao.Voo'] == 'Realizado'].copy()
df_clean = df_clean.dropna(subset=cols_datas + ['distancia_km'])

# D. Features Calculadas
df_clean['delay_minutes'] = (df_clean['Partida.Real'] - df_clean['Partida.Prevista']).dt.total_seconds() / 60
df_clean['duration_minutes'] = (df_clean['Chegada.Real'] - df_clean['Partida.Real']).dt.total_seconds() / 60

# E. Outliers (Mantendo lógica de consistência)
mask_clean = (df_clean['duration_minutes'] > 0) & (df_clean['delay_minutes'] > -60) & (df_clean['delay_minutes'] < 1440)
df_clean = df_clean[mask_clean].copy()

# F. Target (> 15 min)
df_clean['target'] = np.where(df_clean['delay_minutes'] > 15, 1, 0)

# G. Feriados e Tempo
print(" Calculando feriados...")
br_holidays = holidays.Brazil()
df_clean['data_voo'] = df_clean['Partida.Prevista'].dt.date
df_clean['is_holiday'] = df_clean['data_voo'].apply(lambda x: 1 if x in br_holidays else 0)

df_clean['hora'] = df_clean['Partida.Prevista'].dt.hour
df_clean['dia_semana'] = df_clean['Partida.Prevista'].dt.dayofweek
df_clean['mes'] = df_clean['Partida.Prevista'].dt.month

# Renomear para o padrão
df_clean.rename(columns={'Companhia.Aerea': 'companhia', 'Aeroporto.Origem': 'origem', 'Aeroporto.Destino': 'destino'}, inplace=True)

# 4. SPLIT E ENCODING (CORRIGIDO - SEM DATA LEAKAGE)
print(" Realizando Split e Encoding Seguro...")

cols_base = ['companhia', 'origem', 'destino', 'distancia_km', 'hora', 'dia_semana', 'mes', 'is_holiday']
X = df_clean[cols_base]
y = df_clean['target']

# Split ANTES do Encoding
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encoding
encoders = {}
cat_features = ['companhia', 'origem', 'destino']

for col in cat_features:
    le = SafeLabelEncoder()
    # Fit apenas no Treino
    X_train.loc[:, f'{col}_encoded'] = le.fit_transform(X_train[col])
    # Transform no Teste
    X_test.loc[:, f'{col}_encoded'] = le.transform(X_test[col])
    encoders[col] = le

# Seleção final de features
features_finais = [
    'companhia_encoded', 'origem_encoded', 'destino_encoded', 
    'distancia_km', 'hora', 'dia_semana', 'mes', 'is_holiday'
]

X_train_final = X_train[features_finais]
X_test_final = X_test[features_finais]

# 5. TREINAMENTO
print(f" Treinando CatBoost Classifier...")
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)

model.fit(X_train_final, y_train)

# 6. VALIDAÇÃO
probs = model.predict_proba(X_test_final)[:, 1]
preds = (probs >= 0.40).astype(int) # Threshold de negócio
recall = recall_score(y_test, preds)
acc = accuracy_score(y_test, preds)

print(f" Resultado Final -> Recall: {recall:.1%} | Acurácia: {acc:.1%}")

# 7. EXPORTAR
print(" Salvando artefatos de produção...")
artifact = {
    'model': model,
    'encoders': encoders,
    'features': features_finais,
    'metadata': {
        'autor': 'Time Data Science',
        'versao': '3.0.0-CAT',
        'tecnologia': 'CatBoost + SafeEncoder',
        'threshold_recomendado': 0.40,
        'recall_atrasos': recall 
    }
}

joblib.dump(artifact, model_path)
print(f"✅ Arquivo gerado com sucesso: {model_path}")