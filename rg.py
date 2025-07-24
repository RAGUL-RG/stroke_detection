from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import os

# Configure Gemini
genai.configure(api_key="AIzaSyBl0uePdOuYxYwVYxiXpVQ-Smr0HNXv-mA")
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Load and preprocess training data
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

@app.get("/predict")
def predict(
    sex: str,
    age: float,
    hypertension: str,
    heart_disease: str,
    ever_married: str,
    work_type: str,
    residence_type: str,
    avg_glucose_level: float,
    bmi: float,
    smoking_status: str
):
    input_data = {
        "sex": sex,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }

    processed = preprocess_input(input_data)
    input_scaled = scaler.transform(processed)
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = int(model.predict(input_scaled)[0])

    if avg_glucose_level > 180 or bmi > 38:
        prediction = 1

    reasons = []
    flags = []

    if age > 60:
        reasons.append("Age above 60")
    if hypertension.lower() == "yes":
        reasons.append("Has hypertension")
    if heart_disease.lower() == "yes":
        reasons.append("Has heart disease")
    if avg_glucose_level > 180:
        reasons.append("Very high average glucose level (>180 mg/dL)")
        flags.append("High Glucose")
    if bmi > 35:
        reasons.append("Very high BMI (>35)")
        flags.append("High BMI")
    if smoking_status.lower() == "smokes":
        reasons.append("Smoker")


    recommendations = []
    if reasons:
        prompt = f"The patient has the following risk factors: {', '.join(reasons)}. Please summarize and provide personalized, practical health recommendations in bullet points (not more than 10 points)."
        gemini_response = model_gemini.generate_content(prompt)
        gemini_recommendation = gemini_response.text.strip()
        recommendations = [gemini_recommendation]

    return {
        "prediction": prediction,
        "probability": round(probability, 3),
        "reasons": reasons,
        "recommendations": recommendations,
        "flags": flags
    }
