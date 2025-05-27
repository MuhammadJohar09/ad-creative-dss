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

log_path = "logs/prediction_logs.csv"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

def log_prediction(input_dict, prediction_label, preference=None):
    input_dict["prediction"] = prediction_label
    input_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if preference:
        input_dict["preference"] = preference
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=input_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(input_dict)

explainer = shap.TreeExplainer(model)
analyzer = SentimentIntensityAnalyzer()

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

# UI
st.title("AI-Driven Ad Creative Optimization DSS 🚀")
mode = st.radio("Select Mode", ["Single Prediction", "A/B Testing"])

if mode == "Single Prediction":
    img = st.file_uploader("Upload Ad Image", type=["jpg", "jpeg", "png"])
    caption = st.text_area("Enter Caption")
    platform = st.selectbox("Platform", ["Facebook", "Instagram"])
    preference = st.radio("Would you like to submit your preference?", ["Yes", "No"])

    if img and caption:
        image = Image.open(img).convert("RGB")
        st.image(image, caption="Uploaded Ad", use_container_width=True)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        brightness = calculate_brightness(img_cv)
        contrast = calculate_contrast(img_cv)
        face_detected = int(detect_faces(img_cv))
        sentiment = get_sentiment(caption)
        cta = int(has_cta(caption))
        caption_len = caption_length(caption)

        features = pd.DataFrame([[sentiment, cta, caption_len, brightness, contrast, face_detected]],
                                columns=["sentiment_score", "contains_cta", "caption_length", "brightness", "contrast", "face_detected"])
        prediction = model.predict(features)[0]
        predicted_label = le.inverse_transform([prediction])[0]

        input_dict = {
            "sentiment_score": sentiment, "contains_cta": cta, "caption_length": caption_len,
            "brightness": brightness, "contrast": contrast, "face_detected": face_detected
        }
        log_prediction(input_dict, predicted_label, preference if preference == "Yes" else None)

        st.subheader(f"📊 Predicted CTR Class: **{predicted_label}**")
        st.write("🔍 Model Features:")
        st.write(features)
        st.write("🔢 Encoded Prediction:", int(prediction))

        shap_values = explainer.shap_values(features)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, features, plot_type="bar", show=False)
        st.pyplot(fig)

elif mode == "A/B Testing":
    st.markdown("### 📊 Upload Two Ads for A/B Testing")
    col1, col2 = st.columns(2)
    with col1:
        img_a = st.file_uploader("Upload Ad A", type=["jpg", "jpeg", "png"], key="a")
        cap_a = st.text_area("Caption A", key="cap_a")
    with col2:
        img_b = st.file_uploader("Upload Ad B", type=["jpg", "jpeg", "png"], key="b")
        cap_b = st.text_area("Caption B", key="cap_b")

    if img_a and cap_a and img_b and cap_b:
        def predict(img_file, cap):
            image = Image.open(img_file).convert("RGB")
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            brightness = calculate_brightness(img_cv)
            contrast = calculate_contrast(img_cv)
            face_detected = int(detect_faces(img_cv))
            sentiment = get_sentiment(cap)
            cta = int(has_cta(cap))
            caption_len = caption_length(cap)
            features = pd.DataFrame([[sentiment, cta, caption_len, brightness, contrast, face_detected]],
                                    columns=["sentiment_score", "contains_cta", "caption_length", "brightness", "contrast", "face_detected"])
            pred = model.predict(features)[0]
            label = le.inverse_transform([pred])[0]
            return label, features

        label_a, features_a = predict(img_a, cap_a)
        label_b, features_b = predict(img_b, cap_b)

        st.markdown(f"### 🔴 Ad A Prediction: **{label_a}**")
        st.write(features_a)
        st.markdown(f"### 🔵 Ad B Prediction: **{label_b}**")
        st.write(features_b)
