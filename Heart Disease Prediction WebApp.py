# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:43:13 2025

@author: b lakshmi sai prasad
"""

import numpy as np
import pickle
import streamlit as st  # Ensure correct aliasing

# Load the saved model
loaded_model = pickle.load(open("C:/Users/b lakshmi sai prasad/Downloads/trained_model.sav", 'rb'))

# Function for Prediction
def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)
    prediction = loaded_model.predict(input_data_as_numpy_array)

    return "The Person is not affected by Heart Disease" if prediction[0] == 0 else "The person is affected by Heart Disease"

# Streamlit Web App
def main():
    st.title("Heart Disease Prediction Web App")  

    # User Input Fields
    age = st.text_input("Age")
    sex = st.text_input("Sex (0 = Female, 1 = Male)")
    cp = st.text_input("Chest Pain Type (0-3)")
    trestbps = st.text_input("Resting Blood Pressure")
    chol = st.text_input("Cholesterol Level")
    fbs = st.text_input("Fasting Blood Sugar (>120 mg/dl, 1 = Yes, 0 = No)")
    restecg = st.text_input("Resting ECG Results (0-2)")
    thalach = st.text_input("Maximum Heart Rate Achieved")
    exang = st.text_input("Exercise Induced Angina (1 = Yes, 0 = No)")
    oldpeak = st.text_input("ST Depression Induced by Exercise")
    slope = st.text_input("Slope of Peak Exercise ST Segment (0-2)")
    ca = st.text_input("Number of Major Vessels (0-3)")
    thal = st.text_input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)")

    diagnosis = ""

    # Prediction Button
    if st.button("Heart Disease Test Result"):
        try:
            # Convert inputs to float and make predictions
            user_input = [float(age), float(sex), float(cp), float(trestbps),
                          float(chol), float(fbs), float(restecg), float(thalach),
                          float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
            diagnosis = heart_disease_prediction(user_input)
        except ValueError:
            diagnosis = "Please enter valid numerical inputs."

    st.success(diagnosis)

if __name__ == "__main__":
    main()
