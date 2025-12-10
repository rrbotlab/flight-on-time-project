import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI(title="FlightOnTime MVP API")

# --- CARGA DEL MODELO ---
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "flight_model_mvp.joblib")

artifacts = None

try:
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    encoders = artifacts['encoders']
    print("✅ Modelo Random Forest MVP cargado correctamente.")
except Exception as e:
    print(f"⚠️ Error cargando modelo: {e}")

class FlightInput(BaseModel):
    companhia: str
    origem: str
    destino: str
    data_partida: str

# Función segura para evitar errores si llega una aerolínea nueva
def safe_encode(encoder, value):
    try:
        return int(encoder.transform([str(value)])[0])
    except:
        return 0 # Si no existe, usamos la clase 0 por defecto

@app.post("/predict")
def predict_flight(flight: FlightInput):
    if not artifacts:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    try:
        # 1. Procesar Fecha
        dt = pd.to_datetime(flight.data_partida)
        
        # 2. Crear DataFrame de entrada (Igual que en el notebook)
        input_data = pd.DataFrame([{
            'companhia_encoded': safe_encode(encoders['companhia'], flight.companhia),
            'origem_encoded': safe_encode(encoders['origem'], flight.origem),
            'destino_encoded': safe_encode(encoders['destino'], flight.destino),
            'hora': dt.hour,
            'dia_semana': dt.dayofweek,
            'mes': dt.month
        }])
        
        # 3. Predicción
        prob = float(model.predict_proba(input_data)[0][1])
        # Usamos 0.5 como corte estándar para MVP
        es_atrasado = bool(prob > 0.5)

        return {
            "atrasado": es_atrasado,
            "probabilidade": round(prob, 4),
            "nivel_risco": "ALTO" if prob > 0.5 else "BAIXO",
            "modelo": "Random Forest MVP v1"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)