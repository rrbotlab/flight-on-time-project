# data-science/src/app.py
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import holidays
import requests
from datetime import datetime
import pytz # <--- NUEVO IMPORT: Para la conversión de Zona Horaria

app = FastAPI(title="FlightOnTime AI Service (V5.0 Live Weather)")

# --- CONFIGURAÇÃO CORS  ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CARGA ROBUSTA ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "flight_classifier_v4.joblib")

model = None
features_list = []
airport_coords = {}
THRESHOLD = 0.35

# Função Haversine
def calculate_distance(lat1, lon1, lat2, lon2):
    r = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# --- FUNÇÃO: FETCH WEATHER ---
def get_live_weather(lat, long, date_time_str):
    try:
        # Aquí ya llegará la hora local corregida gracias a la vacuna en predict()
        dt = pd.to_datetime(date_time_str)
        date_str = dt.strftime('%Y-%m-%d')
        hour = dt.hour
        
        # OpenMeteo Endpoint (timezone=America/Sao_Paulo es clave aquí)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={long}&hourly=precipitation,wind_speed_10m&start_date={date_str}&end_date={date_str}&timezone=America%2FSao_Paulo"
        
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if 'hourly' in data:
                # Si pedimos las 17h local, OpenMeteo (configurado en SaoPaulo) nos da el índice 17.
                precip = data['hourly']['precipitation'][hour]
                wind = data['hourly']['wind_speed_10m'][hour]
                return float(precip), float(wind), "✅ LIVE (OpenMeteo)"
    except Exception as e:
        print(f" Weather API Error: {e}")
    
    return 0.0, 5.0, " Offline/Date Limit"

# Carga do Modelo
if os.path.exists(model_path):
    try:
        print(f" Carregando: {model_path}")
        artifact = joblib.load(model_path)
        model = artifact['model']
        features_list = artifact['features']
        airport_coords = artifact.get('airport_coords', {})
        meta = artifact.get('metadata', {})
        THRESHOLD = meta.get('threshold', 0.35)
        print(f"✅ V5.0 Online | Live Weather Ready")
    except Exception as e:
        print(f" Erro fatal: {e}")
else:
    print(f" Modelo não encontrado.")

class FlightInput(BaseModel):
    companhia: str
    origem: str
    destino: str
    data_partida: str  
    distancia_km: Optional[float] = None 
    precipitation: Optional[float] = None
    wind_speed: Optional[float] = None

@app.post("/predict")
def predict(flight: FlightInput):
    if not model: raise HTTPException(status_code=503, detail="Service Unavailable")
    
    try:
        # ==============================================================================
        # UTC
        # ==============================================================================
        # 1. Parseamos la fecha. Pandas detecta si viene UTC (Z) o Local.
        dt_obj = pd.to_datetime(flight.data_partida)
        
        # 2. Si tiene zona horaria (ej: UTC del backend nuevo), convertimos a Sao Paulo
        if dt_obj.tz is not None:
            dt_obj = dt_obj.tz_convert('America/Sao_Paulo')
            # 3. Quitamos la info de zona para que el resto del código la trate como "Local Pura"
            #    (Así, 17:00 UTC-3 se convierte en simplemente 17:00)
            dt_obj = dt_obj.tz_localize(None)
        
        # 4. Sobrescribimos la fecha en el objeto flight con la hora LOCAL corregida
        flight.data_partida = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
        
        # ==============================================================================
        # Ahora el resto del código piensa que recibió hora local
        # ==============================================================================

        # 1. Distância Automática
        dist_final = flight.distancia_km
        orig_lat, orig_long = 0, 0
        
        if flight.origem in airport_coords:
            orig_lat = airport_coords[flight.origem]['lat']
            orig_long = airport_coords[flight.origem]['long']

        if dist_final is None or dist_final == 0:
            if flight.origem in airport_coords and flight.destino in airport_coords:
                dest = airport_coords[flight.destino]
                dist_final = calculate_distance(orig_lat, orig_long, dest['lat'], dest['long'])
            else:
                dist_final = 800.0

        # 2. CLIMA AUTOMÁTICO
        precip_final = flight.precipitation
        wind_final = flight.wind_speed
        weather_source = "Manual Input"

        if precip_final is None and wind_final is None:
            if orig_lat != 0:
                # Usamos flight.data_partida que YA ESTÁ CORREGIDA a hora local
                p, w, source = get_live_weather(orig_lat, orig_long, flight.data_partida)
                precip_final = p
                wind_final = w
                weather_source = source
            else:
                precip_final, wind_final = 0.0, 5.0
                weather_source = "No Coords Found"
        else:
            precip_final = precip_final if precip_final is not None else 0.0
            wind_final = wind_final if wind_final is not None else 5.0

        # 3. Pipeline ML
        # Usamos dt_obj que ya calculamos arriba (Hora Local)
        dt = dt_obj 
        is_holiday = 1 if dt.date() in holidays.Brazil() else 0
        
        input_df = pd.DataFrame([{
            'companhia': str(flight.companhia),
            'origem': str(flight.origem),
            'destino': str(flight.destino),
            'distancia_km': float(dist_final),
            'hora': dt.hour,      # Ahora será 17 (Local) en vez de 20 (UTC)
            'dia_semana': dt.dayofweek,
            'mes': dt.month,
            'is_holiday': is_holiday,
            'precipitation': float(precip_final),
            'wind_speed': float(wind_final),
            'clima_imputado': 0
        }])
        
        if features_list:
            input_df = input_df[features_list]
        
        prob = float(model.predict_proba(input_df)[0][1])
        
        if prob < THRESHOLD:
            status, color = "🟢 PONTUAL", "green"
        elif prob < 0.70:
            status, color = "🟡 ALERTA", "yellow"
        else:
            status, color = "🔴 ATRASO PROVÁVEL", "red"
            
        return {
            "previsao": status,
            "probabilidade": round(prob, 4),
            "cor": color,
            "dados_utilizados": {
                "distancia": round(dist_final, 1),
                "chuva": precip_final,
                "vento": wind_final,
                "fonte_clima": weather_source
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc() # Esto ayuda a ver errores en la terminal
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)