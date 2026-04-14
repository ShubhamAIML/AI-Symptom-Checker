from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from google.colab import files

# File path daal
df = pd.read_csv('/content/drive/My Drive/DiseaseAndSymptoms.csv')
print(df.head())  # Check karo

# Define Symptom columns
symptom_columns = [f"Symptom_{i}" for i in range(1, 18)]

# Conver Symptoms into list
df["Symptoms"] = df[symptom_columns].apply(lambda row: [x for x in row if pd.notna(x)], axis=1)

# Encode Symptoms Through MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])
y = df["Disease"]  # Pehla column Disease hai
print(X.shape)  # Dimensions check karo

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                              class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Accuracy check
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Save Model aur encoder
joblib.dump(model, 'symptom_checker_model.pkl')
joblib.dump(mlb, 'mlb_encoder.pkl')

# Download files
files.download('symptom_checker_model.pkl')
files.download('mlb_encoder.pkl')
print("Model and encoder downloaded! Download CSV manually from Drive.")
