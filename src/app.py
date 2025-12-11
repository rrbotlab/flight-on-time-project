import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(title="FlightOnTime AI Service (V3)")

# --- 1. CARGA DE ARTEFACTOS ---
MODEL_FILENAME = "flight_classifier_mvp.joblib"
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, MODEL_FILENAME)

artifacts = None

try:
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    encoders = artifacts['encoders']
    # Recuperamos las features para asegurar el orden
    expected_features = artifacts.get('features', [
        'companhia_encoded', 'origem_encoded', 'destino_encoded', 
        'distancia_km', 'hora', 'dia_semana', 'mes'
    ])
    print(f"✅ Modelo V3 cargado exitosamente.")
except Exception as e:
    print(f"⚠️ Error cargando modelo: {e}")

# --- 2. INPUT ---
class FlightInput(BaseModel):
    companhia: str
    origem: str
    destino: str
    data_partida: str
    distancia_km: float

# --- 3. HELPER ---
def safe_encode(encoder, value):
    try:
        return int(encoder.transform([str(value)])[0])
    except:
        return 0 

# --- 4. ENDPOINT INTELIGENTE ---
@app.post("/predict")
def predict_flight(flight: FlightInput):
    if not artifacts:
        raise HTTPException(status_code=500, detail="Modelo no disponible")

    try:
        # A. Preparar Datos
        dt = pd.to_datetime(flight.data_partida)
        
        input_data = pd.DataFrame([{
            'companhia_encoded': safe_encode(encoders['companhia'], flight.companhia),
            'origem_encoded': safe_encode(encoders['origem'], flight.origem),
            'destino_encoded': safe_encode(encoders['destino'], flight.destino),
            'distancia_km': float(flight.distancia_km),
            'hora': dt.hour,
            'dia_semana': dt.dayofweek,
            'mes': dt.month
        }])
        
        # Ordenar columnas
        input_data = input_data[expected_features]
        
        # B. Predicción (Probabilidad)
        prob = float(model.predict_proba(input_data)[0][1])
        
        # C. Lógica de Semáforo (Threshold Tuning 0.40)
        # Basado en análisis de sensibilidad (Recall 86%)
        if prob < 0.40:
            status = "PONTUAL"
            risco = "BAIXO"
            msg = "Voo com boas condições operacionais."
        elif 0.40 <= prob < 0.60:
            status = "ALERTA" # Zona gris
            risco = "MEDIO"
            msg = "Risco moderado. Monitorar painel."
        else: # > 0.60
            status = "ATRASADO"
            risco = "ALTO"
            msg = "Alta probabilidade de atraso (>15 min)."

        # D. Respuesta Enriquecida
        return {
            "previsao": status,
            "probabilidade": round(prob, 4),
            "nivel_risco": risco,
            "mensagem": msg,
            "detalles": {
                "distancia": flight.distancia_km,
                "hora_partida": dt.hour
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)