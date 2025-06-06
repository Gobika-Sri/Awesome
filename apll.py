import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and prepare data
df = pd.read_csv('dataset.csv')
df.fillna('None', inplace=True)

symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
df['Symptoms'] = df[symptom_cols].values.tolist()

all_symptoms = sorted({sym.strip() for symptoms in df["Symptoms"] for sym in symptoms if sym.strip() != "None"})
symptom_index = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

def encode_symptoms(symptom_list):
    vector = np.zeros(len(all_symptoms), dtype=int)
    for symptom in symptom_list:
        if symptom in symptom_index:
            vector[symptom_index[symptom]] = 1
    return vector

X = np.array([encode_symptoms(s) for s in df["Symptoms"]])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Disease'])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title("Smart Disease Predictor")

st.markdown(f"*Model Accuracy:* {accuracy:.2%}")

selected_symptoms = st.multiselect("Select your symptoms:", all_symptoms)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_vector = encode_symptoms(selected_symptoms).reshape(1, -1)
        prediction = model.predict(input_vector)
        disease = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🧬 Predicted Disease: *{disease}*")