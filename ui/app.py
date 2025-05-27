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

# Set main title
st.title("AI-Driven Ad Creative Optimization DSS 🚀")

# Load model and label encoder
model = xgb.XGBClassifier()
model.load_model("models/xgb_ctr_model.json")
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Log predictions
log_path = "logs/prediction_logs.csv"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
def log_prediction(input_dict, prediction_label):
    input_dict["prediction"] = prediction_label
    input_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(log_path)
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=input_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(input_dict)

# Feature extraction functions
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

# Mode selection: Single or A/B Testing
mode = st.radio("Choose prediction mode:", ["🎯 Single Creative", "🆚 A/B Testing"])

if mode == "🎯 Single Creative":
    uploaded_image = st.file_uploader("Upload Ad Image", type=["jpg", "jpeg", "png"])
    caption = st.text_area("Enter Caption")
    platform = st.selectbox("Choose Platform", ["Facebook", "Instagram"])
    if uploaded_image and caption:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Ad", use_container_width=True)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        brightness = calculate_brightness(img_cv)
        contrast = calculate_contrast(img_cv)
        face_detected = int(detect_faces(img_cv))
        sentiment_score = get_sentiment(caption)
        contains_cta = int(has_cta(caption))
        caption_len = caption_length(caption)
        features = pd.DataFrame([[sentiment_score, contains_cta, caption_len, brightness, contrast, face_detected]],
                                columns=["sentiment_score", "contains_cta", "caption_length", "brightness", "contrast", "face_detected"])
        prediction = model.predict(features)[0]
        predicted_label = le.inverse_transform([prediction])[0]
        st.subheader(f"📊 Predicted CTR Class: **{predicted_label}**")
        st.write(features)
        st.write(int(prediction))
        st.write(predicted_label)
        log_prediction({"creative": "Single", "caption": caption, "platform": platform}, predicted_label)
        with st.expander("🔍 Explanation (SHAP)"):
            fig, ax = plt.subplots()
            shap.plots.bar(
                shap.Explanation(values=explainer.shap_values(features)[0],
                                 base_values=explainer.expected_value,
                                 data=features.values[0],
                                 feature_names=features.columns.tolist()),
                show=False
            )
            st.pyplot(fig)
        if predicted_label == "High":
            st.success("🎉 This ad is predicted to perform well!")
        else:
            st.warning("⚠️ This ad might need improvement for better CTR.")

elif mode == "🆚 A/B Testing":
    st.subheader("Upload Creative A")
    image_a = st.file_uploader("Creative A Image", type=["jpg", "jpeg", "png"], key="image_a")
    caption_a = st.text_area("Creative A Caption", key="caption_a")
    st.subheader("Upload Creative B")
    image_b = st.file_uploader("Creative B Image", type=["jpg", "jpeg", "png"], key="image_b")
    caption_b = st.text_area("Creative B Caption", key="caption_b")
    platform = st.selectbox("Choose Platform", ["Facebook", "Instagram"])
    user_preference = st.radio("Choose your preferred creative:", ["A", "B"])
    if st.button("🔍 Predict CTR for A/B Testing"):
        if image_a and image_b and caption_a and caption_b:
            img_a = Image.open(image_a).convert("RGB")
            img_b = Image.open(image_b).convert("RGB")
            img_cv_a = cv2.cvtColor(np.array(img_a), cv2.COLOR_RGB2BGR)
            img_cv_b = cv2.cvtColor(np.array(img_b), cv2.COLOR_RGB2BGR)
            features_a = pd.DataFrame([[get_sentiment(caption_a), int(has_cta(caption_a)), caption_length(caption_a),
                                        calculate_brightness(img_cv_a), calculate_contrast(img_cv_a), int(detect_faces(img_cv_a))]],
                                      columns=["sentiment_score", "contains_cta", "caption_length", "brightness", "contrast", "face_detected"])
            features_b = pd.DataFrame([[get_sentiment(caption_b), int(has_cta(caption_b)), caption_length(caption_b),
                                        calculate_brightness(img_cv_b), calculate_contrast(img_cv_b), int(detect_faces(img_cv_b))]],
                                      columns=["sentiment_score", "contains_cta", "caption_length", "brightness", "contrast", "face_detected"])
            pred_a = model.predict(features_a)[0]
            pred_b = model.predict(features_b)[0]
            label_a = le.inverse_transform([pred_a])[0]
            label_b = le.inverse_transform([pred_b])[0]
            log_prediction({"creative": "A", "caption": caption_a, "platform": platform}, label_a)
            log_prediction({"creative": "B", "caption": caption_b, "platform": platform}, label_b)
            st.subheader("🔍 Prediction Results")
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**Creative A CTR:** {label_a}")
                st.image(img_a, use_container_width=True)
                figA, axA = plt.subplots()
                shap.plots.bar(
                    shap.Explanation(values=explainer.shap_values(features_a)[0],
                                     base_values=explainer.expected_value,
                                     data=features_a.values[0],
                                     feature_names=features_a.columns.tolist()),
                    show=False
                )
                st.pyplot(figA)
            with colB:
                st.markdown(f"**Creative B CTR:** {label_b}")
                st.image(img_b, use_container_width=True)
                figB, axB = plt.subplots()
                shap.plots.bar(
                    shap.Explanation(values=explainer.shap_values(features_b)[0],
                                     base_values=explainer.expected_value,
                                     data=features_b.values[0],
                                     feature_names=features_b.columns.tolist()),
                    show=False
                )
                st.pyplot(figB)
            st.markdown(f"👍 **Your preference:** Creative {user_preference}")
            better_creative = "A" if label_a == "High" else "B" if label_b == "High" else "A" if pred_a > pred_b else "B"
            st.success(f"🏆 Recommended creative based on prediction: **Creative {better_creative}**")

