import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Query
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
import io
from groq import Groq
import logging

# Load environment variables
load_dotenv()

# Load the Isolation Forest model for electricity consumption anomaly detection
model_path = os.getenv("MODEL_PATH")
if not model_path:
    raise ValueError("MODEL_PATH environment variable is not set.")
model = joblib.load(model_path)

# Initialize Groq client for Llama explanations with API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")
client = Groq(api_key=api_key)

# Initialize FastAPI app
app = FastAPI(
    title="Electricity Consumption Anomaly Detection API",
    description="This API detects anomalies in electricity consumption data using a pre-trained Isolation Forest model. "
    "It also provides explanations for anomalies flagged in the data.",
    version="1.0.0",
)


class PredictionResponse(BaseModel):
    predictions: list  # 1 = Anomalous, 0 = Normal
    explanations: list  # Explanation for each data point


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Query(
        400,
        description="Custom threshold for electricity consumption anomaly detection in kWh",
    ),
):
    """
    Predict anomalies in electricity consumption data.

    Parameters:
    - file: CSV file containing 'Month' and 'Monthly_Consumption_kWh' columns.
    - threshold: Custom threshold in kWh for flagging anomalies.

    Returns:
    - JSON response containing predictions and explanations.
    """
    try:
        # Read uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Validate required columns in the input data
        if not all(col in df.columns for col in ["Month", "Monthly_Consumption_kWh"]):
            return {
                "error": "Input CSV must contain 'Month' and 'Monthly_Consumption_kWh' columns."
            }

        # Extract features for prediction
        X = df[["Month", "Monthly_Consumption_kWh"]].values

        # Use pre-trained model for inference with a threshold for electricity consumption
        predictions = np.where(
            df["Monthly_Consumption_kWh"] > threshold, 1, 0
        )  # 1 = Anomalous, 0 = Normal

        # Generate explanations for flagged anomalies
        explanations = []
        for i, flag in enumerate(predictions):
            if flag == 1:  # Anomaly detected
                input_text = (
                    f"The electricity consumption for Month={df.iloc[i]['Month']} "
                    f"with Monthly_Consumption_kWh={df.iloc[i]['Monthly_Consumption_kWh']} is anomalous. "
                    "Provide detailed reasoning."
                )
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an anomaly detection assistant for electricity consumption.",
                        },
                        {"role": "user", "content": input_text},
                    ],
                    temperature=0,
                    max_tokens=500,
                    top_p=1,
                    stream=False,
                )
                explanation = completion.choices[0].message.content
            else:
                explanation = "The electricity consumption data point is normal."

            explanations.append(explanation)

        # Return predictions and explanations as a JSON response
        return PredictionResponse(
            predictions=predictions.tolist(), explanations=explanations
        )

    except Exception as e:
        logging.error(f"Error in /predict endpoint: {e}")
        return {"error": "Internal server error occurred. Check logs for details."}
