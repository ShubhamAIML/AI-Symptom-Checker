from flask import Flask, request, render_template, flash, session, redirect, url_for
import numpy as np
import joblib
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'x7k9p2m4q8r5t1n3j6h0'  # Secret key for flash messages and session

# Load model, encoder, and CSV locally
model = joblib.load("symptom_checker_model.pkl")
mlb = joblib.load("mlb_encoder.pkl")
df = pd.read_csv("DiseaseAndSymptoms.csv")

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
    feedback_submitted = session.get('feedback_submitted', False)  # Success flag
    feedback_error = session.get('feedback_error', False)  # Error flag

    if request.method == "POST" and 'symptom-form' in request.form:
        for category in symptom_categories:
            selected_symptoms.extend(request.form.getlist(category))
        
        if len(selected_symptoms) < 3:
            return render_template("index.html", categories=symptom_categories, 
                                 selected_symptoms=selected_symptoms, 
                                 error="Please select at least 3 symptoms",
                                 feedback_submitted=feedback_submitted,
                                 feedback_error=feedback_error)
        
        # Binary vector banao
        input_vector = np.zeros(len(mlb.classes_))
        for symptom in selected_symptoms:
            if symptom in mlb.classes_:
                input_vector[np.where(mlb.classes_ == symptom)] = 1

        # Predict karo with probabilities
        probabilities = model.predict_proba([input_vector])[0]
        top_preds = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]
        prediction = top_preds[0][0]
        others = [f"{p[0]} ({p[1] * 100:.0f}%)" for p in top_preds[1:]]  # Percentage format mein
        
        # Session mein prediction aur others store karo
        session['prediction'] = prediction
        session['others'] = others
        
        return redirect(url_for('result'))  # Redirect to result page
    
    # Feedback form handling
    if request.method == "POST" and 'feedback-form' in request.form:
        name = request.form.get('name')
        email = request.form.get('email')
        feedback = request.form.get('feedback')
        
        if not name or not email or not feedback:
            session['feedback_error'] = True  # Error flag set karo
            return redirect(url_for('home'))  # Redirect karo
        else:
            # Current date aur time dd/mm/yy format mein
            submit_time = datetime.now().strftime("%d/%m/%y %H:%M:%S")
            # Feedback ko file mein save karo with date-time
            with open("feedback.txt", "a") as f:
                f.write(f"Date: {submit_time}, Name: {name}, Email: {email}, Feedback: {feedback}\n")
            session['feedback_submitted'] = True  # Success flag set karo
            print(f"Feedback received - Date: {submit_time}, Name: {name}, Email: {email}, Feedback: {feedback}")
            return redirect(url_for('home'))  # Redirect karo taaki repeat na ho
    
    # Feedback flags ko reset karo jab page normally load ho
    if request.method == "GET":
        if 'feedback_submitted' in session:
            session.pop('feedback_submitted')
        if 'feedback_error' in session:
            session.pop('feedback_error')

    return render_template("index.html", categories=symptom_categories, 
                         selected_symptoms=selected_symptoms, 
                         feedback_submitted=feedback_submitted,
                         feedback_error=feedback_error)

@app.route("/result")
def result():
    # Session se prediction aur others retrieve karo
    prediction = session.get('prediction')
    others = session.get('others', [])
    
    if not prediction:
        return redirect(url_for('home'))  # Agar session mein data nahi hai to home pe redirect
    
    return render_template("result.html", prediction=prediction, others=others)

if __name__ == "__main__":
    app.run(debug=True)
