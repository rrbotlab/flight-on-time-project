# back-end/app.py
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import holidays

app = FastAPI(title="FlightOnTime AI Service (V4.1 Native)")

# Carga do Modelo
MODEL_FILENAME = "../data-science/src/flight_classifier_v4.joblib"
# Nota: Ajuste o caminho se necessário dependendo de onde rodar o uvicorn
if not os.path.exists(MODEL_FILENAME):
    # Tenta caminho local se rodar de dentro da pasta src
    MODEL_FILENAME = "flight_classifier_v4.joblib"

try:
    artifact = joblib.load(MODEL_FILENAME)
    model = artifact['model']
    features_list = artifact['features']
    meta = artifact.get('metadata', {})
    THRESHOLD = meta.get('threshold', 0.35)
    print(f"✅ Modelo V4.1 Carregado! Threshold: {THRESHOLD}")
except Exception as e:
    model = None
    print(f" Erro ao carregar modelo: {e}")

class FlightInput(BaseModel):
    companhia: str
    origem: str
    destino: str
    data_partida: str  
    distancia_km: float
    precipitation: float = 0.0
    wind_speed: float = 5.0

@app.post("/predict")
def predict(flight: FlightInput):
    if not model: return {"error": "Model not loaded"}
    
    dt = pd.to_datetime(flight.data_partida)
    is_holiday = 1 if dt.date() in holidays.Brazil() else 0
    
    # DataFrame direto (Strings puras para o CatBoost)
    input_df = pd.DataFrame([{
        'companhia': str(flight.companhia),
        'origem': str(flight.origem),
        'destino': str(flight.destino),
        'distancia_km': float(flight.distancia_km),
        'hora': dt.hour,
        'dia_semana': dt.dayofweek,
        'mes': dt.month,
        'is_holiday': is_holiday,
        'precipitation': float(flight.precipitation),
        'wind_speed': float(flight.wind_speed),
        'clima_imputado': 0
    }])
    
    # Garantir ordem das colunas
    input_df = input_df[features_list]
    
    # Predição
    prob = float(model.predict_proba(input_df)[0][1])
    
    # Semáforo V4.1
    if prob < THRESHOLD:
        status, color = "🟢 PONTUAL", "green"
    elif prob < 0.70:
        status, color = "🟡 ALERTA CLIMÁTICO/OPERACIONAL", "yellow"
    else:
        status, color = "🔴 ALTA PROBABILIDADE DE ATRASO", "red"
        
    return {
        "previsao": status,
        "probabilidade": round(prob, 4),
        "cor": color,
        "clima": f"Chuva: {flight.precipitation}mm"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)