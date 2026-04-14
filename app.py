from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('heart_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = int(data.get('age', 40))
    symptoms = data.get('symptoms', [])
    notes = data.get('notes', "").lower()

    # Model Input Mapping
    cp = 1 if "Chest Pain" in symptoms else 0
    bp = 150 if "High BP" in symptoms else 120
    chol = 260 if "Fatigue" in symptoms else 200
    features = np.array([[age, cp, bp, chol]])

    # Prediction
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    # Keyword Audit
    danger_words = ['pain', 'attack', 'tightness', 'history', 'diabetes', 'stroke']
    is_emergency = any(word in notes for word in danger_words)

    # Radar Data Logic
    radar = [min(100, (age/85)*100), min(100, (len(symptoms)/6)*100), 
             100 if is_emergency else 20, 90 if cp else 30, 80 if "High BP" in symptoms else 40]

    # Clinical Result Logic
    if prediction == 1 or is_emergency or len(symptoms) >= 3:
        res = {
            "score": int(prob * 100), "risk": "High", "organ": "Coronary Arteries",
            "disease": "Acute Coronary Syndrome (ACS)", "color": "#f87171",
            "tips": [
                "Immediate Cardiology Consultation required for Troponin-T testing.",
                "Schedule a 12-Lead Electrocardiogram (ECG) & Echocardiogram.",
                "Administer 325mg Aspirin (under medical supervision) if ischemia is suspected.",
                "Perform Comprehensive Lipid Profile & HbA1c screening.",
                "Absolute physical rest; avoid all strenuous metabolic activity."
            ], "radar_data": radar
        }
    elif len(symptoms) > 0:
        res = {
            "score": int(prob * 100), "risk": "Medium", "organ": "Circulatory System",
            "disease": "Hypertensive Heart Disease", "color": "#fbbf24",
            "tips": [
                "Initiate 24-hour Ambulatory Blood Pressure Monitoring (ABPM).",
                "Reduce Sodium intake to < 1500mg/day (DASH Diet protocol).",
                "Screen for secondary causes of hypertension (Renal Artery Stenosis).",
                "Increase aerobic physical activity to 150 mins per week.",
                "Monitor Fasting Blood Glucose & Electrolyte balance."
            ], "radar_data": radar
        }
    else:
        res = {
            "score": 15, "risk": "Low", "organ": "Healthy Myocardium",
            "disease": "Clinical Normality", "color": "#34d399",
            "tips": [
                "Maintain Mediterranean dietary pattern for cardiovascular longevity.",
                "Continue routine biometric screening every 12 months.",
                "Focus on stress management through mindful meditation.",
                "Maintain BMI between 18.5 - 24.9.",
                "Stay hydrated and maintain optimal Vitamin D3 levels."
            ], "radar_data": radar
        }

    return jsonify(res)

if __name__ == '__main__':
    app.run(debug=True)