import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('dataset.csv')

# Fill NaNs and normalize symptom columns
df.fillna('None', inplace=True)
symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]

# Combine symptoms into a list per row with cleaning
df['Symptoms'] = df[symptom_cols].values.tolist()
df['Symptoms'] = df['Symptoms'].apply(lambda x: [sym.strip().lower() for sym in x if sym.strip().lower() != 'none'])

# Create unique sorted list of all symptoms
all_symptoms = sorted({sym for symptoms in df["Symptoms"] for sym in symptoms})
symptom_index = {symptom: idx for idx, symptom in enumerate(all_symptoms)}

# Function to encode input symptoms into binary vector
def encode_symptoms(symptom_list):
    vector = np.zeros(len(all_symptoms), dtype=int)
    for symptom in symptom_list:
        symptom = symptom.strip().lower()
        if symptom in symptom_index:
            vector[symptom_index[symptom]] = 1
    return vector

# Encode full dataset
X = np.array([encode_symptoms(s) for s in df["Symptoms"]])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Disease"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.set_page_config(page_title="Smart Disease Predictor", layout="centered")
st.title("🧠 Smart Disease Predictor")
st.markdown("#### Select your symptoms below:")

# Show model accuracy
st.markdown(f"<span style='color:green'><b>Model Accuracy:</b> {accuracy:.2%}</span>", unsafe_allow_html=True)

# Symptom selection
selected_symptoms = st.multiselect("Choose your symptoms:", all_symptoms)

# Prediction
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        input_vector = encode_symptoms(selected_symptoms).reshape(1, -1)
        prediction = model.predict(input_vector)
        disease = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🧬 Predicted Disease: **{disease}**")
