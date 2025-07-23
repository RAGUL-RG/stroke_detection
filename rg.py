from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model_gemini = genai.GenerativeModel("gemini-1.5-flash")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class StrokeInput(BaseModel):
    sex: str
    age: float
    hypertension: str
    heart_disease: str
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str
def preprocess_input(data: dict):
    sex_map = {"male": 1, "female": 0}
    hypertension_map = {"yes": 1, "no": 0}
    heart_disease_map = {"yes": 1, "no": 0}
    married_map = {"yes": 1, "no": 0}
    residence_map = {"urban": 1, "rural": 0}
    smoking_map = {"smokes": 1, "never": 0}
    work_map = {
        "never_worked": 0,
        "children": 1,
        "govt_job": 2,
        "self_employed": 3,
        "private": 4
    }

    return [[
        sex_map.get(data["sex"].lower(), 0),
        float(data["age"]),
        hypertension_map.get(data["hypertension"].lower(), 0),
        heart_disease_map.get(data["heart_disease"].lower(), 0),
        married_map.get(data["ever_married"].lower(), 0),
        work_map.get(data["work_type"].lower(), 4),
        residence_map.get(data["residence_type"].lower(), 0),
        float(data["avg_glucose_level"]),
        float(data["bmi"]),
        smoking_map.get(data["smoking_status"].lower(), 0),
    ]]


train_df = pd.read_csv("test.csv")
train_df = train_df.fillna(train_df.mean(numeric_only=True))

features = [
    "sex", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "residence_type",
    "avg_glucose_level", "bmi", "smoking_status"
]
X = train_df[features]
y = train_df["stroke"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


@app.post("/predict")
def predict(data: StrokeInput):
    input_data = data.dict()
    processed = preprocess_input(input_data)
    input_scaled = scaler.transform(processed)

    probability = model.predict_proba(input_scaled)[0][1]
    prediction = int(model.predict(input_scaled)[0])

    if input_data["avg_glucose_level"] > 180 or input_data["bmi"] > 38:
        prediction = 1

    reasons = []
    flags = []

    if input_data["age"] > 60:
        reasons.append("Age above 60")
    if input_data["hypertension"].lower() == "yes":
        reasons.append("Has hypertension")
    if input_data["heart_disease"].lower() == "yes":
        reasons.append("Has heart disease")
    if input_data["avg_glucose_level"] > 180:
        reasons.append("Very high glucose level")
        flags.append("High Glucose")
    if input_data["bmi"] > 35:
        reasons.append("Very high BMI")
        flags.append("High BMI")
    if input_data["smoking_status"].lower() == "smokes":
        reasons.append("Smoker")
    if reasons:
        prompt = f"The patient has the following risk factors: {', '.join(reasons)}. Provide personalized and practical health recommendations."
        gemini_response = model_gemini.generate_content(prompt)
        final_recommendations = gemini_response.text.strip()
    else:
        final_recommendations = "No major health risk factors found. Maintain a healthy lifestyle!"

    return {
        "prediction": prediction,
        "probability": round(probability, 3),
        "reasons": reasons,
        "flags": flags,
        "recommendations": final_recommendations
    }
