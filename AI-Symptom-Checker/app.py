from flask import Flask, request, render_template
import numpy as np
import joblib
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'x7k9p2m4q8r5t1n3j6h0'  # Random secret key (production mein secure key use karo)

# Load model, encoder, and CSV locally
model = joblib.load(os.path.join(os.path.dirname(__file__), "symptom_checker_model.pkl"))
mlb = joblib.load(os.path.join(os.path.dirname(__file__), "mlb_encoder.pkl"))
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "DiseaseAndSymptoms.csv"))

# Symptom columns define karo
symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]

# Saare symptoms extract karo
all_symptoms = sorted(set([symptom for col in symptom_columns for symptom in df[col].dropna().unique()]))

# Symptom categories dynamically banayein (except 'Other')
symptom_categories = {
    "Skin": [s for s in all_symptoms if any(kw in s for kw in ["skin", "rash", "itch", "patch", "eruption"])],
    "Respiratory": [s for s in all_symptoms if any(kw in s for kw in ["cough", "breath", "sputum", "chest", "phlegm"])],
    "Digestive": [s for s in all_symptoms if any(kw in s for kw in ["vomit", "nausea", "abdominal", "diarrhoea", "constipation", "ulcer", "acidity"])],
    "General": [s for s in all_symptoms if any(kw in s for kw in ["fever", "fatigue", "chill", "sweat", "malaise", "weight", "thirst"])],
    "Neurological": [s for s in all_symptoms if any(kw in s for kw in ["headache", "dizz", "balance", "confusion", "numb"])]
}

# 'Other' category ko baaki categories ke baad calculate karo
symptom_categories["Other"] = [s for s in all_symptoms if s not in sum(symptom_categories.values(), [])]

@app.route("/", methods=["GET", "POST"])
def home():
    selected_symptoms = []
    if request.method == "POST":
        for category in symptom_categories:
            selected_symptoms.extend(request.form.getlist(category))
        
        if len(selected_symptoms) < 3:
            return render_template("index.html", categories=symptom_categories, 
                                 selected_symptoms=selected_symptoms, 
                                 error="Please select at least 3 symptoms")
        
        # Binary vector banao
        input_vector = np.zeros(len(mlb.classes_))
        for symptom in selected_symptoms:
            if symptom in mlb.classes_:
                input_vector[np.where(mlb.classes_ == symptom)] = 1

        # Predict karo with probabilities
        probabilities = model.predict_proba([input_vector])[0]
        top_preds = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]
        prediction = top_preds[0][0]
        others = [f"{p[0]} ({p[1]:.2f})" for p in top_preds[1:]]
        
        return render_template("result.html", prediction=prediction, others=others)
    
    return render_template("index.html", categories=symptom_categories, selected_symptoms=selected_symptoms)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
