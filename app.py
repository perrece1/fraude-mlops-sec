from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import logging

# Configurar logs para ver qué pasa en producción de forma segura
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = joblib.load('modelo_fraude.pkl')
    logger.info("Modelo cargado con éxito en la API!")
except Exception as e:
    logger.error(f"Error crítico al cargar el modelo: {e}")
    raise RuntimeError("No se pudo cargar el modelo de fraude.")

app = FastAPI(
    title="API de Detección de Fraude (MLOps Sec)",
    description="API para predecir fraudes en transacciones de tarjetas de crédito usando RandomForest.",
    version="1.0.0"
)

class TransactionData(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def home():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_fraud(data: TransactionData):
    try:
        input_data = data.model_dump()
        df_input = pd.DataFrame([input_data])
        
        prediction = model.predict(df_input)
        probabilities = model.predict_proba(df_input)
        fraud_probability = float(probabilities[0][1])
        
        return {
            "fraude": bool(prediction[0] == 1),
            "probabilidad_fraude": round(fraud_probability, 4)
        }
        
    except Exception as e:
        # Registramos el error real en los logs internos del contenedor
        logger.error(f"Error durante la predicción: {str(e)}")
        # Al cliente externo le devolvemos una respuesta limpia y sin trazas de código
        raise HTTPException(status_code=500, detail="Error interno al procesar la predicción.")

if __name__ == "__main__":
    # Desactivamos el reload para el despliegue de producción seguro
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)