import os
import csv
from datetime import datetime
import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import xgboost as xgb
import shap
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle

# Load model and label encoder
model = xgb.XGBClassifier()
model.load_model("./models/xgb_ctr_model.json")

with open("./models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Create logs directory if it doesn't exist
log_path = "logs/prediction_logs.csv"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# Function to log predictions
def log_prediction(input_dict, prediction_label):
    input_dict["prediction"] = prediction_label
    input_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=input_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(input_dict)

# SHAP setup
explainer = shap.TreeExplainer(model)

# Sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Feature extraction functions
def calculate_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])

def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

def get_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

def has_cta(text):
    cta_keywords = ["buy", "shop", "order", "book", "now", "get", "click", "learn more", "sign up", "call"]
    return any(kw in text.lower() for kw in cta_keywords)

def caption_length(text):
    return len(str(text))

# Streamlit UI
st.title("AI-Driven Ad Creative Optimization DSS 🚀")
uploaded_image = st.file_uploader("Upload Ad Image", type=["jpg", "jpeg", "png"])
caption = st.text_area("Enter Caption")
platform = st.selectbox("Choose Platform", ["Facebook", "Instagram"])

if uploaded_image and caption:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Ad", use_container_width=True)  # updated from use_column_width

    # Preprocess
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    brightness = calculate_brightness(img_cv)
    contrast = calculate_contrast(img_cv)
    face_detected = int(detect_faces(img_cv))

    # Text features
    sentiment_score = get_sentiment(caption)
    contains_cta = int(has_cta(caption))
    caption_len = caption_length(caption)

    features = pd.DataFrame([[sentiment_score, contains_cta, caption_len, brightness, contrast, face_detected]],
                            columns=["sentiment_score", "contains_cta", "caption_length", "brightness", "contrast", "face_detected"])

    prediction = model.predict(features)[0]
    predicted_label = le.inverse_transform([prediction])[0]

    # Logging
    input_dict = {
        "sentiment_score": sentiment_score,
        "contains_cta": contains_cta,
        "caption_length": caption_len,
        "brightness": brightness,
        "contrast": contrast,
        "face_detected": face_detected
    }
    log_prediction(input_dict, predicted_label)

    # Display Results
    st.subheader(f"📊 Predicted CTR Class: **{predicted_label}**")
    st.subheader("🔍 Model Input Features")
    st.write(features)
    st.subheader("🔢 Raw Prediction (Encoded)")
    st.write(int(prediction))
    st.subheader("🎯 Predicted CTR Label")
    st.write(predicted_label)

    # SHAP Explanation (summary bar plot)
    shap_values = explainer.shap_values(features)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    st.pyplot(fig)
