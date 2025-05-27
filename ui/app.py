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
import matplotlib.pyplot as plt

# Load model and label encoder
model = xgb.XGBClassifier()
model.load_model("models/xgb_ctr_model.json")
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# SHAP setup
explainer = shap.TreeExplainer(model)
analyzer = SentimentIntensityAnalyzer()

# Utility functions
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

def predict_ctr(image, caption):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    brightness = calculate_brightness(img_cv)
    contrast = calculate_contrast(img_cv)
    face_detected = int(detect_faces(img_cv))
    sentiment_score = get_sentiment(caption)
    contains_cta = int(has_cta(caption))
    caption_len = caption_length(caption)
    features = pd.DataFrame([[sentiment_score, contains_cta, caption_len, brightness, contrast, face_detected]],
                            columns=["sentiment_score", "contains_cta", "caption_length", "brightness", "contrast", "face_detected"])
    pred = model.predict(features)[0]
    shap_vals = explainer.shap_values(features)
    return pred, shap_vals, features

def log_prediction(input_dict, prediction_label):
    input_dict["prediction"] = prediction_label
    input_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = "logs/prediction_logs.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=input_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(input_dict)

# Streamlit UI
st.title("🚀 AI-Driven Ad Creative Optimization DSS")

# Single Creative Prediction Section
st.header("📢 Predict CTR for a Single Ad Creative")
single_image = st.file_uploader("Upload Your Ad Creative", type=["jpg", "jpeg", "png"])
single_caption = st.text_area("Enter Caption for Your Creative")
platform = st.selectbox("Select Platform", ["Facebook", "Instagram"], key="platform_single")

if single_image and single_caption:
    if st.button("🎯 Predict CTR for Single Creative"):
        img = Image.open(single_image).convert("RGB")
        st.image(img, caption="Uploaded Creative", use_column_width=True)
        pred, shap_vals, features = predict_ctr(img, single_caption)
        label = le.inverse_transform([pred])[0]
        
        log_prediction({"creative":"Single", "caption":single_caption, "platform":platform}, label)
        
        st.subheader(f"📊 Predicted CTR Class: **{label}**")
        st.write(features)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_vals, features, plot_type="bar", show=False)
        st.pyplot(fig)

# A/B Testing Section
st.header("📸 A/B Testing for Creative Optimization")
col1, col2 = st.columns(2)
with col1:
    image_a = st.file_uploader("Upload Creative A", type=["jpg", "jpeg", "png"], key="A")
    caption_a = st.text_area("Caption for Creative A", key="caption_A")
with col2:
    image_b = st.file_uploader("Upload Creative B", type=["jpg", "jpeg", "png"], key="B")
    caption_b = st.text_area("Caption for Creative B", key="caption_B")

if image_a and image_b and caption_a and caption_b:
    user_choice = st.radio("💡 Which creative do you prefer?", ("Creative A", "Creative B"))
    if st.button("🎯 Predict CTR and Recommend Best Creative"):
        img_a = Image.open(image_a).convert("RGB")
        pred_a, shap_a, features_a = predict_ctr(img_a, caption_a)
        label_a = le.inverse_transform([pred_a])[0]
        
        img_b = Image.open(image_b).convert("RGB")
        pred_b, shap_b, features_b = predict_ctr(img_b, caption_b)
        label_b = le.inverse_transform([pred_b])[0]
        
        log_prediction({"creative":"A", "caption":caption_a, "platform":platform}, label_a)
        log_prediction({"creative":"B", "caption":caption_b, "platform":platform}, label_b)
        
        st.subheader("🔍 Prediction Results")
        colA, colB = st.columns(2)
        with colA:
            st.image(img_a, caption=f"Creative A\nPredicted CTR: {label_a}")
            st.write(features_a)
            st.write(f"Predicted Label: {label_a}")
            figA, axA = plt.subplots()
            shap.summary_plot(shap_a, features_a, plot_type="bar", show=False)
            st.pyplot(figA)
        with colB:
            st.image(img_b, caption=f"Creative B\nPredicted CTR: {label_b}")
            st.write(features_b)
            st.write(f"Predicted Label: {label_b}")
            figB, axB = plt.subplots()
            shap.summary_plot(shap_b, features_b, plot_type="bar", show=False)
            st.pyplot(figB)
        
        priority = {"High": 3, "Medium": 2, "Low": 1}
        best_creative = "A" if priority[label_a] > priority[label_b] else "B"
        st.success(f"🌟 Recommended Best Creative: **Creative {best_creative}**")
        st.info(f"👤 Your preference: **{user_choice}**")
