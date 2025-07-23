from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

train_df = pd.read_csv("test.csv", encoding="latin1", on_bad_lines="skip",engine="python")
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

model = LogisticRegression()
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
    prediction = int(model.predict(input_scaled)[0])

    # Rule-based explanation and advice
    reasons = []
    recommendations = []

    if age > 60:
        reasons.append("Age above 60")
        recommendations.append("Maintain regular checkups and a healthy lifestyle.")
    if hypertension.lower() == "yes":
        reasons.append("Has hypertension")
        recommendations.append("Reduce salt, manage stress, and take prescribed meds.")
    if heart_disease.lower() == "yes":
        reasons.append("Has heart disease")
        recommendations.append("Consult a cardiologist and follow heart-healthy routines.")
    if avg_glucose_level > 180:
        reasons.append("High average glucose level")
        recommendations.append("Control blood sugar through diet, exercise, and medication.")
    if bmi > 30:
        reasons.append("High BMI (overweight)")
        recommendations.append("Adopt a calorie-conscious diet and regular physical activity.")
    if smoking_status.lower() == "smokes":
        reasons.append("Smoker")
        recommendations.append("Seek help to quit smoking and avoid tobacco.")

    return {
        "prediction": prediction,
        "reasons": reasons if prediction == 1 else [],
        "recommendations": recommendations if prediction == 1 else []
    }
