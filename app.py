import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set the page configuration (must be at the very beginning of the script)
st.set_page_config(page_title="Diabetes Risk Prediction App", layout="wide")

# Correct file paths using raw string literals (r"...")
DATA_URL = r"C:\Users\u-art\OneDrive\Desktop\diabetes project\resampled_diabetes_data.csv"
MODEL_PATH = r"C:\Users\u-art\OneDrive\Desktop\diabetes project\trained_rf_model.pkl"

# Debugging: Print paths to verify correctness
st.write(f"Data path: {DATA_URL}")
st.write(f"Model path: {MODEL_PATH}")

# Function to load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)  # Ensure the file exists at this location
        st.write("Dataset loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to load the model
@st.cache_resource
def load_model():
    try:
        rf_model = joblib.load(MODEL_PATH)  # Ensure the model file exists at this location
        st.write("Model loaded successfully!")
        return rf_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load data and model
df = load_data()
rf_model = load_model()

if df is None or rf_model is None:
    st.stop()  # If there is an error loading the data or model, stop the app

# Extract feature names (excluding the target variable 'Outcome')
feature_names = df.drop(columns=['Outcome']).columns.tolist()

# Function to predict diabetes outcome
def predict_outcome(model, input_data):
    """Make a prediction given user input."""
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit UI Enhancements
st.title("🔬 Diabetes Risk Prediction App")
st.write("Enter patient data below to predict the likelihood of diabetes.")

# Sidebar Information
st.sidebar.header("ℹ️ About This App")
st.sidebar.write("This app uses a trained Random Forest model to predict diabetes based on patient data.")
st.sidebar.write("🔹 Model: Random Forest Classifier")
st.sidebar.write("🔹 Data Source: Processed Diabetes Dataset")

# User Input Form in Two Columns
col1, col2 = st.columns(2)
input_values = {}

# Dynamic user inputs with feature names
for i, feature in enumerate(feature_names):
    col = col1 if i % 2 == 0 else col2
    # Provide a default value for DiabetesPedigreeFunction if the user doesn't have it
    if feature == "DiabetesPedigreeFunction":
        # Default to the mean or zero if the user doesn't enter a value
        default_value = df[feature].mean()  # or use 0 for a default value
        input_values[feature] = col.number_input(f"{feature} (optional)", value=float(default_value))
    else:
        input_values[feature] = col.number_input(f"{feature}", value=float(df[feature].mean()))

# Add Family History Checkbox
family_history = st.radio("Do you have a family history of diabetes?", ('No', 'Yes'))

# Adjust DiabetesPedigreeFunction based on family history
if family_history == 'Yes':
    input_values['DiabetesPedigreeFunction'] = 0.5  # Set to a specific value for family history
else:
    input_values['DiabetesPedigreeFunction'] = 0  # If no family history, set to 0

# Predict button with better UI
if st.button("🔍 Predict Diabetes Outcome", use_container_width=True):
    with st.spinner("Making prediction..."):
        result = predict_outcome(rf_model, input_values)
        outcome = "Diabetic" if result >= 0.5 else "Non-Diabetic"
        st.success(f"🩺 Predicted Outcome: **{outcome}**")

        # Display result in a styled metric box
        st.metric(label="Diabetes Risk Score", value=round(result, 2))

        # Display an additional visual indicator of the risk
        if result >= 0.5:
            st.warning("⚠️ High Risk of Diabetes! Please consult a doctor.")
        else:
            st.success("✅ Low Risk of Diabetes!")

# Footer Information
st.write("---")
st.write("👨‍⚕️ **Disclaimer:** This prediction is based on machine learning and should not replace professional medical advice.")
st.write("📅 **Last Updated:** March 2025")
