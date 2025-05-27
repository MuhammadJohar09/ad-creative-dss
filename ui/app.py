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

# Create folder if it doesn't exist
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
explainer = shap.Explainer(model)

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

def predict_ctr(image, caption):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    brightness = calculate_brightness(img_cv)
    contrast = calculate_contrast(img_cv)
    face_detected = int(detect_faces(img_cv))
    sentiment_score = get_sentiment(caption)
    contains_cta = int(has_cta(caption))
    caption_len = caption_length(caption)

    features = pd.DataFrame([[
        sentiment_score, contains_cta, caption_len,
        brightness, contrast, face_detected
    ]], columns=[
        "sentiment_score", "contains_cta", "caption_length",
        "brightness", "contrast", "face_detected"
    ])
    prediction = model.predict(features)[0]
    shap_values = explainer(features)
    return prediction, shap_values, features

# Streamlit UI
st.title("AI-Driven Ad Creative Optimization DSS")
mode = st.radio("Select Mode", ["📷 Single Creative Prediction", "🆚 A/B Testing"])

if mode == "📷 Single Creative Prediction":
    uploaded_image = st.file_uploader("Upload Ad Image", type=["jpg", "jpeg", "png"])
    caption = st.text_area("Enter Caption")
    platform = st.selectbox("Choose Platform", ["Facebook", "Instagram"], key="platform_single")

    if uploaded_image and caption:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Ad", use_container_width=True)

        prediction, shap_values, features = predict_ctr(img, caption)
        predicted_label = le.inverse_transform([prediction])[0]

        log_prediction({"creative": "Single", "caption": caption, "platform": platform}, predicted_label)

        st.subheader(f"📊 Predicted CTR Class: **{predicted_label}**")
        st.write("\n")
        if predicted_label == "High":
            st.success("✅ This ad is likely to perform well!")
        else:
            st.warning("⚠️ This ad might need improvement to increase CTR.")

        with st.expander("🔍 Explanation (SHAP)"):
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values[0], show=False)
            st.pyplot(fig)

elif mode == "🆚 A/B Testing":
    col1, col2 = st.columns(2)
    with col1:
        image_a = st.file_uploader("Upload Creative A", type=["jpg", "jpeg", "png"], key="image_a")
        caption_a = st.text_area("Enter Caption A", key="caption_a")
    with col2:
        image_b = st.file_uploader("Upload Creative B", type=["jpg", "jpeg", "png"], key="image_b")
        caption_b = st.text_area("Enter Caption B", key="caption_b")

    if image_a and image_b and caption_a and caption_b:
        platform_ab = st.selectbox("Select Platform for A/B Testing", ["Facebook", "Instagram"], key="platform_ab")
        user_choice = st.radio("💡 Which creative do you prefer?", ("Creative A", "Creative B"))
        if st.button("🎯 Predict CTR and Recommend Best Creative"):
            img_a = Image.open(image_a).convert("RGB")
            pred_a, shap_a, features_a = predict_ctr(img_a, caption_a)
            label_a = le.inverse_transform([pred_a])[0]

            img_b = Image.open(image_b).convert("RGB")
            pred_b, shap_b, features_b = predict_ctr(img_b, caption_b)
            label_b = le.inverse_transform([pred_b])[0]

            log_prediction({"creative":"A", "caption":caption_a, "platform":platform_ab}, label_a)
            log_prediction({"creative":"B", "caption":caption_b, "platform":platform_ab}, label_b)

            st.subheader("🔍 Prediction Results")
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**Creative A CTR:** {label_a}")
                figA, axA = plt.subplots()
                shap.plots.bar(shap_a[0], show=False)
                st.pyplot(figA)
            with colB:
                st.markdown(f"**Creative B CTR:** {label_b}")
                figB, axB = plt.subplots()
                shap.plots.bar(shap_b[0], show=False)
                st.pyplot(figB)

            st.subheader("✅ Final Recommendation")
            if label_a == "High" and label_b != "High":
                st.success("✅ Recommend using Creative A")
            elif label_b == "High" and label_a != "High":
                st.success("✅ Recommend using Creative B")
            elif label_a == "High" and label_b == "High":
                st.info("Both creatives are likely to perform well!")
            else:
                st.warning("⚠️ Neither ad is predicted to have High CTR. Consider revising creatives.")
