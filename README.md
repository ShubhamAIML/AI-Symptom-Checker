# 🚀 AI: Intelligent Symptom Checker

**Predict diseases with AI-powered accuracy!** 🌐 [Try it now](https://ai-symptom-checker-j93h.onrender.com/)

---

## 🌟 Key Features

- **Smart Symptom Analysis**: Instant disease predictions based on selected symptoms  
- **Organized Categories**: Symptoms grouped into Skin, Respiratory, Digestive, General, Neurological, and Other  
- **Multiple Predictions**: Top 3 probable diseases with confidence scores  
- **User Feedback System**: Share feedback to improve model performance  
- **Secure & Private**: No permanent storage of personal data  

---

## 💻 Tech Stack

| Category           | Technology                           |
|--------------------|--------------------------------------|
| **Backend**        | Python (Flask)                       |
| **Machine Learning** | Scikit-learn (RandomForestClassifier) |
| **Data Processing** | Pandas, NumPy                        |
| **Frontend**       | HTML, CSS, JavaScript (Bootstrap)    |
| **Model Persistence** | Joblib                             |

---

## 🧠 Technical Details

### Machine Learning Model

- **Algorithm**: RandomForestClassifier (Scikit-learn)  
- **Why Random Forest?**  
  - Handles high-dimensional, sparse binary data efficiently  
  - Robust to noise and imbalanced datasets  
  - Provides probability scores for multi-class classification  
- **Training Data**: `DiseaseAndSymptoms.csv`  
  - Maps diseases to up to 17 symptoms per disease  
  - Symptoms encoded as binary features using MultiLabelBinarizer (MLB)  
- **Preprocessing**:  
  - Symptoms categorized into Skin, Respiratory, Digestive, General, Neurological, and Other  
  - Binary input vector created for model inference  
- **Model Serialization**:  
  - Model saved as `symptom_checker_model.pkl` using Joblib  
  - MultiLabelBinarizer saved as `mlb_encoder.pkl`  
- **Prediction Output**:  
  - Top 3 disease predictions with probability scores (e.g., "Disease X (85%)")  
  - Probabilities computed using `predict_proba`  

### Backend

- **Framework**: Flask (lightweight Python web framework)  
- **Functionality**:  
  - Handles GET/POST requests for symptom input and feedback submission  
  - Session management for storing prediction results  
  - Renders `index.html` and `result.html` templates  
- **Security**:  
  - Secret key for session management and flash messages  
  - No permanent storage of sensitive user data  

### Frontend

- **Templates**:  
  - `index.html`: Symptom selection form with categorized checkboxes  
  - `result.html`: Displays primary and secondary predictions  
- **Styling**: Bootstrap for responsive, modern UI  
- **Interactivity**: JavaScript for dynamic form handling  

### Feedback System

- Stores feedback in `feedback.txt` with timestamp, name, email, and comments  
- Uses session flags to prevent duplicate submissions  

---

## 🛠 Quick Setup

### Prerequisites

- Python 3.8+  
- Virtual environment (recommended)  
- Git  

### Clone & Install

```bash
git clone https://github.com/yourusername/Al-Symptom-Checker.git
cd Al-Symptom-Checker
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Required Files

- `DiseaseAndSymptoms.csv`: Dataset mapping diseases to symptoms  
- `symptom_checker_model.pkl`: Pre-trained RandomForest model  
- `mlb_encoder.pkl`: Pre-trained MultiLabelBinarizer  

### Run the App

```bash
python app.py
```

🔗 **Open in browser**: [http://localhost:5000](http://localhost:5000)  

---

## 📂 Project Structure

```
Al-Symptom-Checker/
├── app.py                    # Flask backend
├── templates/                # Frontend pages
│   ├── index.html            # Symptom selection page
│   └── result.html           # Prediction results page
├── DiseaseAndSymptoms.csv    # Symptom-disease dataset
├── symptom_checker_model.pkl # Trained RandomForest model
├── mlb_encoder.pkl           # MultiLabelBinarizer encoder
├── feedback.txt              # Stores user feedback
└── requirements.txt          # Python dependencies
```

---

## 🔍 How It Works

1. **Select Symptoms**: Choose at least 3 symptoms from categorized lists  
2. **AI Prediction**: RandomForestClassifier predicts top 3 diseases using binary symptom vector  
3. **View Results**: See primary prediction and secondary predictions with confidence scores  
4. **Give Feedback**: Submit feedback to improve the model  

---

## 📈 Why Use This?

- **Fast & Accurate**: Trained on real symptom-disease data with robust RandomForest model  
- **User-Friendly**: Intuitive UI with categorized symptoms and clear results  
- **Privacy-First**: No unnecessary data collection; feedback stored anonymously  

---

## 🚀 Future Upgrades

- **Doctor Consultation Integration**: Connect users to medical professionals  
- **Multi-Language Support**: Expand accessibility globally  
- **Mobile App Version**: Native iOS/Android apps  
- **Symptom Severity Analysis**: Incorporate symptom intensity for refined predictions  

---

## 📜 License

MIT License - Free for personal and commercial use  

---

## 💬 Feedback & Support

- **Love it?** Star the repo! ⭐  
- **Found a bug?** Open an issue! 🐞  
- **Suggestions?** Let’s make it better! 💡  

**Let’s revolutionize healthcare with AI!** 🚑💡  

---

## 🔗 Get Started Now!

👉 `git clone https://github.com/yourusername/Al-Symptom-Checker.git`  

---
