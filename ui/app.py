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
model.load_model("C:/Users/johar/.jupyter/ad-creative-dss/models/xgb_ctr_model.json")

with open("C:/Users/johar/.jupyter/ad-creative-dss/models/label_encoder.pkl", "rb") as f:
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

def extract_features(image, caption):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    brightness = calculate_brightness(img_cv)
    contrast = calculate_contrast(img_cv)
    face_detected = int(detect_faces(img_cv))
    sentiment_score = get_sentiment(caption)
    contains_cta = int(has_cta(caption))
    caption_len = caption_length(caption)
    return pd.DataFrame([{
        "sentiment_score": sentiment_score,
        "contains_cta": contains_cta,
        "caption_length": caption_len,
        "brightness": brightness,
        "contrast": contrast,
        "face_detected": face_detected
    }]), {
        "sentiment_score": sentiment_score,
        "contains_cta": contains_cta,
        "caption_length": caption_len,
        "brightness": brightness,
        "contrast": contrast,
        "face_detected": face_detected,
        "caption": caption
    }

# Streamlit UI
st.title("AI-Powered Ad CTR Predictor 🚀")

mode = st.radio("Select Mode", ["Single Prediction", "A/B Testing"])

if mode == "Single Prediction":
    uploaded_image = st.file_uploader("Upload Ad Image", type=["jpg", "jpeg", "png"])
    caption = st.text_area("Enter Caption")
    platform = st.selectbox("Choose Platform", ["Facebook", "Instagram"])

    if uploaded_image and caption:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Ad", use_column_width=True)

        features, input_dict = extract_features(img, caption)

        prediction = model.predict(features)[0]
        predicted_label = le.inverse_transform([prediction])[0]

        input_dict["platform"] = platform
        log_prediction(input_dict, predicted_label)

        st.subheader(f"📊 Predicted CTR Class: **{predicted_label}**")

        st.subheader("🔍 Model Input Features")
        st.write(features)

        st.subheader("🔢 Raw Prediction (Encoded)")
        st.write(int(prediction))

        st.subheader("🎯 Predicted CTR Label")
        st.write(predicted_label)

        shap_values = explainer.shap_values(features)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, features, plot_type="bar", show=False)
        st.pyplot(fig)

elif mode == "A/B Testing":
    st.markdown("## 🔀 A/B Testing: Compare Two Ad Creatives")
    col1, col2 = st.columns(2)

    with col1:
        st.header("Ad Creative A")
        image_a = st.file_uploader("Upload Image A", type=["jpg", "jpeg", "png"], key="a_img")
        caption_a = st.text_area("Caption A", key="caption_a")

    with col2:
        st.header("Ad Creative B")
        image_b = st.file_uploader("Upload Image B", type=["jpg", "jpeg", "png"], key="b_img")
        caption_b = st.text_area("Caption B", key="caption_b")

    user_choice = st.radio("Which ad do you prefer?", ("Ad Creative A", "Ad Creative B"))

    if st.button("Submit Preference"):
        if image_a and caption_a and image_b and caption_b:
            img_a = Image.open(image_a).convert("RGB")
            img_b = Image.open(image_b).convert("RGB")

            features_a, dict_a = extract_features(img_a, caption_a)
            features_b, dict_b = extract_features(img_b, caption_b)

            pred_a = le.inverse_transform([model.predict(features_a)[0]])[0]
            pred_b = le.inverse_transform([model.predict(features_b)[0]])[0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Creative A")
                st.image(img_a, use_column_width=True)
                st.write("Predicted CTR:", pred_a)

            with col2:
                st.subheader("Creative B")
                st.image(img_b, use_column_width=True)
                st.write("Predicted CTR:", pred_b)

            st.success(f"✅ You selected: **{user_choice}**")

            # Optional: log A/B results
            dict_a["predicted"] = pred_a
            dict_b["predicted"] = pred_b
            dict_a["user_selected"] = user_choice
            dict_b["user_selected"] = user_choice
            log_prediction(dict_a, pred_a)
            log_prediction(dict_b, pred_b)
        else:
            st.warning("Please upload both images and enter both captions before submitting.")
