import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request  # To download model file

# ✅ Ensure set_page_config is the first Streamlit command
st.set_page_config(page_title="Diabetes Risk Prediction App", layout="wide")

# ✅ Correct URLs for data & model
DATA_URL = "https://raw.githubusercontent.com/cashbooda/diabetes-ml-project/main/resampled_diabetes_data.csv"
MODEL_URL = "https://raw.githubusercontent.com/cashbooda/diabetes-ml-project/main/trained_rf_model.pkl"

# ✅ Function to load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        st.write("✅ Dataset loaded successfully!")
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

# ✅ Function to load the model from GitHub
@st.cache_resource
def load_model():
    try:
        model_path = "trained_rf_model.pkl"  # Save file locally
        urllib.request.urlretrieve(MODEL_URL, model_path)  # Download from GitHub
        rf_model = joblib.load(model_path)  # Load the model
        st.write("✅ Model loaded successfully!")
        return rf_model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ✅ Load data and model
df = load_data()
rf_model = load_model()

# Stop execution if data or model fails to load
if df is None or rf_model is None:
    st.stop()

# ✅ Extract feature names (excluding the target variable 'Outcome')
feature_names = df.drop(columns=['Outcome']).columns.tolist()

# ✅ Function to make predictions
def predict_outcome(model, input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

# 🎨 **Streamlit UI Enhancements**
st.title("🔬 Diabetes Risk Prediction App")
st.write("Enter patient data below to predict the likelihood of diabetes.")

# 📝 **Sidebar Information**
st.sidebar.header("ℹ️ About This App")
st.sidebar.write("This app uses a trained **Random Forest model** to predict diabetes.")
st.sidebar.write("🔹 Model: Random Forest Classifier")
st.sidebar.write("🔹 Data Source: Processed Diabetes Dataset")

# ✅ **User Input Form (Two Columns)**
col1, col2 = st.columns(2)
input_values = {}

for i, feature in enumerate(feature_names):
    col = col1 if i % 2 == 0 else col2
    
    # ✅ Make "DiabetesPedigreeFunction" optional
    if feature == "DiabetesPedigreeFunction":
        default_value = df[feature].mean()  # Default to mean or 0
        input_values[feature] = col.number_input(f"{feature} (optional)", value=float(default_value), min_value=0.0)
    else:
        input_values[feature] = col.number_input(f"{feature}", value=float(df[feature].mean()))

# ✅ **Add Family History Question**
family_history = st.radio("Do you have a family history of diabetes?", ('No', 'Yes'))

# ✅ Adjust "DiabetesPedigreeFunction" Based on Family History
if family_history == 'Yes':
    input_values['DiabetesPedigreeFunction'] = 0.5  # Higher default value for family history
else:
    input_values['DiabetesPedigreeFunction'] = 0  # No family history

# 🎯 **Prediction Button**
if st.button("🔍 Predict Diabetes Outcome", use_container_width=True):
    with st.spinner("Making prediction..."):
        result = predict_outcome(rf_model, input_values)
        outcome = "Diabetic" if result >= 0.5 else "Non-Diabetic"
        st.success(f"🩺 **Predicted Outcome: {outcome}**")

        # 🎯 **Display Result as a Metric**
        st.metric(label="Diabetes Risk Score", value=round(result, 2))

        # 🚨 **Show Risk Warning**
        if result >= 0.5:
            st.warning("⚠️ High Risk of Diabetes! Please consult a doctor.")
        else:
            st.success("✅ Low Risk of Diabetes!")

# 📌 **Footer**
st.write("---")
st.write("👨‍⚕️ **Disclaimer:** This prediction is based on machine learning and should not replace professional medical advice.")
st.write("📅 **Last Updated:** March 2025 - By Krish Jain")
