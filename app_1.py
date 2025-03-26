from fastapi import FastAPI, File, UploadFile
from io import StringIO
import pandas as pd
from joblib import load


app = FastAPI()


@app.get("/health")
def health_check():
    return {"status":"OK"}


@app.post("/predict")
async def predict_house(file : UploadFile = File(...)):
    classifier = load('linear_regression.joblib')

    features_df = pd.read_csv("selected_features.csv")
    features = features_df["0"].tolist()

    content = await file.read()

    df = pd.read_csv(StringIO(content.decode('utf-8')))
    df = df[features]
    
    predictions = classifier.predict(df)

    return {
        "predictions": predictions.tolist()
    }
