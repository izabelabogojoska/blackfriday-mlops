from fastapi import FastAPI
import pandas as pd

from model_loader import load_model
from pipeline.features import build_features

app = FastAPI()
model = load_model()

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    features = build_features(df)
    prediction = model.predict(features)

    return {"abuse": bool(prediction[0])}
