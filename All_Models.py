# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 03:27:41 2024

@author: PANDEY
"""

import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import load

# Function to preprocess and predict on a new image
def predict_new_image(model, scaler, image):
    
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance brown spots by adjusting the intensity values
    enhanced_image = cv2.addWeighted(img_gray, 1.5, np.zeros_like(img_gray), 0, 50)
    
    # Resize the image
    new_img_resized = cv2.resize(enhanced_image, (300, 300))

    # Extract HOG features for the new image
    new_img_hog = hog(new_img_resized, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), block_norm='L2-Hys')

    # Standardize the features
    new_img_feature = scaler.transform(np.array([new_img_hog]))

    # Predict using the loaded model
    prediction = model.predict(new_img_feature)

    return prediction[0]

# Load the pre-trained models and scaler
model_names = ['Linear Kernel', 'Poly Kernel', 'RBF Kernel', 'Sigmoid Kernel', 'Ensemble Voting']  

selected_model_name = st.selectbox("Select a model:", model_names)

if selected_model_name == 'Linear Kernel':
    loaded_model = load('linear.joblib')
elif selected_model_name == 'Poly Kernel':
    loaded_model = load('poly.joblib')
elif selected_model_name == 'RBF Kernel':
    loaded_model = load('rbf.joblib')
elif selected_model_name == 'Sigmoid Kernel':
    loaded_model = load('sigmoid.joblib')
elif selected_model_name == 'Ensemble Voting':
    loaded_model = load('voting.joblib')
else:
    st.error("Invalid model selection.")

scaler = load('rice_scaler.joblib')

# Streamlit App
st.title("Rice Disease Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Predict using the selected model
    prediction = predict_new_image(loaded_model, scaler, image)

    st.write(f"Predicted Class ({selected_model_name}): {prediction}")
