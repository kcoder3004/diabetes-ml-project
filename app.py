import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request
import os

# Set page config (must be first)
st.set_page_config(page_title="Diabetes Risk Prediction App", layout="wide")

# Correct data & model paths
DATA_URL = "https://raw.githubusercontent.com/cashbooda/diabetes-ml-project/main/resampled_diabetes_data.csv"
MODEL_URL = "https://raw.githubusercontent.com/cashbooda/diabetes-ml-project/main/trained_rf_model.pkl"

# Function to load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        st.write("✅ Dataset loaded successfully!")
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

# Function to download and load model
@st.cache_resource
def load_model():
    try:
        model_filename = "trained_rf_model.pkl"

        # Download model if not already downloaded
        if not os.path.exists(model_filename):
            urllib.request.urlretrieve(MODEL_URL, model_filename)

        # Load the model
        rf_model = joblib.load(model_filename)
        st.write("✅ Model loaded successfully!")
        return rf_model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load data and model
df = load_data()
rf_model = load_model()

if df is None or rf_model is None:
    st.stop()  # Stop app if loading fails

# Feature names for input
feature_names = df.drop(columns=['Outcome']).columns.tolist()

# Prediction function
def predict_outcome(model, input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit UI
st.title("🔬 Diabetes Risk Prediction App")
st.write("Enter patient data below to predict the likelihood of diabetes.")

# Sidebar Info
st.sidebar.header("ℹ️ About")
st.sidebar.write("GlucoPredict")
st.sidebar.write("Uses a Random Forest model to predict diabetes.")

# Input Form
col1, col2 = st.columns(2)
input_values = {}

for i, feature in enumerate(feature_names):
    col = col1 if i % 2 == 0 else col2
    if feature == "DiabetesPedigreeFunction":
        input_values[feature] = col.number_input(f"{feature} (optional)", value=float(df[feature].mean()))
    else:
        input_values[feature] = col.number_input(f"{feature}", value=float(df[feature].mean()))

# Family history option
family_history = st.radio("Do you have a family history of diabetes?", ('No', 'Yes'))
input_values['DiabetesPedigreeFunction'] = 0.5 if family_history == 'Yes' else 0

# Prediction button
if st.button("🔍 Predict Diabetes Outcome", use_container_width=True):
    with st.spinner("Making prediction..."):
        result = predict_outcome(rf_model, input_values)
        outcome = "Diabetic" if result >= 0.5 else "Non-Diabetic"
        st.success(f"🩺 Predicted Outcome: **{outcome}**")
        st.metric(label="Diabetes Risk Score", value=round(result, 2))

        if result >= 0.5:
            st.warning("⚠️ High Risk of Diabetes! Please consult a doctor.")
        else:
            st.success("✅ Low Risk of Diabetes!")

# Footer
st.write("---")
st.write("👨‍⚕️ **Disclaimer:** This is a machine learning prediction and should not replace medical advice.")
st.write("📅 **Last Updated:** March 2025 - By Krish Jain")
