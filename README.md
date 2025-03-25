# Diabetes Risk Prediction App

## Inspiration
Diabetes is a major global health issue affecting millions of people. Early detection and risk assessment can help individuals take proactive measures to manage their health. This project leverages machine learning to predict diabetes risk based on user-inputted health data.

## What It Does
The **Diabetes Risk Prediction App** is a web-based tool that allows users to input their health metrics and receive a prediction of their diabetes risk. It uses a trained **Random Forest Classifier** to analyze patient data and provide insights into their likelihood of having diabetes.

## How We Built It
- **Data Preprocessing**: Cleaned and resampled diabetes data to balance the dataset.
- **Feature Engineering**: Selected relevant features, handled missing values, and normalized inputs.
- **Model Training**: Used a **Random Forest Classifier** trained on the processed dataset.
- **Web Deployment**: Integrated the model with **Streamlit** to create an interactive UI.
- **GitHub & Streamlit Cloud**: Managed version control and deployed the app online.

## Challenges We Ran Into
- Handling missing or optional values like **DiabetesPedigreeFunction**.
- Streamlit deployment issues due to missing **requirements.txt**.
- Ensuring a user-friendly interface for non-technical users.

## Accomplishments That We're Proud Of
- Successfully trained a machine learning model with **high accuracy**.
- Built an interactive and intuitive **web app** for diabetes prediction.
- Implemented a **family history option** to make predictions more user-friendly.
- Hosted the application on **Streamlit Cloud** for easy accessibility.

## What We Learned
- The importance of **data preprocessing** for improving model performance.
- Best practices for **deploying ML models** in real-world applications.
- Streamlit and GitHub workflows for seamless cloud deployment.

## What's Next for Diabetes Tester
- **Enhancing Model Accuracy**: Experimenting with deep learning models.
- **Expanding Feature Set**: Incorporating additional health indicators.
- **Mobile App Integration**: Making the prediction tool available on smartphones.
- **Continuous Model Training**: Updating the model as new data becomes available.

## Built With
- **Python** – Core language for model development.
- **Pandas & NumPy** – Data handling and processing.
- **Scikit-Learn** – Machine learning model training.
- **Streamlit** – Web application framework for deployment.
- **GitHub** – Version control and project collaboration.
- **Joblib** – Model serialization and loading.

## Created by
**[Your Name / Team Name]**

## My Contribution
✅ **Data Preprocessing & Feature Engineering** – Cleaning, transforming, and resampling diabetes data.
✅ **Machine Learning Model Development** – Training and optimizing a **Random Forest Classifier**.
✅ **Model Deployment with Streamlit** – Creating an interactive web app for diabetes risk prediction.
✅ **GitHub & Streamlit Cloud Integration** – Managing version control and hosting the app online.
✅ **User Experience Enhancements** – Adding a **family history option** and optimizing UI/UX.

---
### 🚀 How to Run the Project
1. Clone the repository:
   ```sh
   git clone https://github.com/cashbooda/diabetes-ml-project.git
   cd diabetes-ml-project
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
4. Open the displayed **localhost URL** in your browser.


---

**Disclaimer:** This tool is for informational purposes only and should not replace professional medical advice. If you have health concerns, consult a healthcare professional.

