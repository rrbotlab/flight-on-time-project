import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import holidays
import catboost
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. DEFINIÇÃO DA CLASSE (Obrigatória para carregar o joblib) ---
class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.classes_ = {}
        self.unknown_token = -1

    def fit(self, y):
        unique_labels = pd.Series(y).unique()
        self.classes_ = {label: idx for idx, label in enumerate(unique_labels)}
        return self

    def transform(self, y):
        return pd.Series(y).apply(lambda x: self.classes_.get(x, self.unknown_token))

app = FastAPI(title="FlightOnTime AI Service (V3 - CatBoost)")

# --- 2. CARGA DE ARTEFATOS ---
MODEL_FILENAME = "flight_classifier_mvp.joblib"
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, MODEL_FILENAME)

artifacts = None
br_holidays = holidays.Brazil()

try:
    print(f" Carregando modelo de: {model_path}")
    artifacts = joblib.load(model_path)
    
    model = artifacts['model']
    encoders = artifacts['encoders']
    expected_features = artifacts.get('features', [])
    metadata = artifacts.get('metadata', {})
    
    THRESHOLD = metadata.get('threshold_recomendado', 0.40)
    
    print(f"✅ Modelo V3 carregado! Versão: {metadata.get('versao')}")
    print(f" Threshold configurado: {THRESHOLD}")

except Exception as e:
    print(f" ERRO CRÍTICO ao carregar modelo: {e}")
    model = None
    THRESHOLD = 0.40

# --- 3. MODELO DE DADOS (INPUT) ---
class FlightInput(BaseModel):
    companhia: str
    origem: str
    destino: str
    data_partida: str
    distancia_km: float

# --- 4. ENDPOINT DE PREDIÇÃO ---
@app.post("/predict")
def predict_flight(flight: FlightInput):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo não carregado no servidor")

    try:
        # A. Processar Data e Feriado
        dt = pd.to_datetime(flight.data_partida)
        
        # .date() garante que a hora não atrapalhe a verificação do feriado
        is_holiday = 1 if dt.date() in br_holidays else 0

        # B. Criar DataFrame base (Dados brutos)
        # Importante: Criamos listas [] para o pandas entender como linha
        input_dict = {
            'companhia': [flight.companhia],
            'origem': [flight.origem],
            'destino': [flight.destino],
            'distancia_km': [flight.distancia_km],
            'hora': [dt.hour],
            'dia_semana': [dt.dayofweek],
            'mes': [dt.month],
            'is_holiday': [is_holiday]
        }
        df_input = pd.DataFrame(input_dict)

        # C. Aplicar Encoders (SafeLabelEncoder)
        # O SafeLabelEncoder vai retornar -1 se a cidade/cia for nova
        for col in ['companhia', 'origem', 'destino']:
            if col in encoders:
                df_input[f'{col}_encoded'] = encoders[col].transform(df_input[col])
            else:
                df_input[f'{col}_encoded'] = -1

        # D. Selecionar features na ordem correta
        X_final = df_input[expected_features]
        
        # E. Predição
        prob = float(model.predict_proba(X_final)[0][1])
        
        # F. Lógica de Semáforo
        if prob < THRESHOLD:
            status = "PONTUAL"
            risco = "BAIXO"
            msg = "Voo com boas condições operacionais."
        elif THRESHOLD <= prob < 0.60:
            status = "ALERTA"
            risco = "MEDIO"
            msg = "Risco moderado. Recomendamos monitorar o status."
        else: # >= 0.60
            status = "ATRASADO"
            risco = "ALTO"
            msg = f"Alta probabilidade de atraso ({prob:.1%}). Planeje-se."

        return {
            "previsao": status,
            "probabilidade": round(prob, 4),
            "nivel_risco": risco,
            "mensagem": msg,
            "detalhes": {
                "is_feriado": bool(is_holiday),
                "distancia_km": flight.distancia_km
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)