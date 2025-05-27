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

explainer = shap.TreeExplainer(model)
analyzer = SentimentIntensityAnalyzer()

def extract_image_features(img_cv):
    brightness = np.mean(cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)[:, :, 2])
    contrast = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY).std()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY), 1.1, 4)
    face_detected = int(len(faces) > 0)
    return brightness, contrast, face_detected

def extract_text_features(caption):
    return (analyzer.polarity_scores(caption)['compound'],
            int(any(kw in caption.lower() for kw in ["buy", "shop", "order", "book", "now", "get", "click", "learn more", "sign up", "call"])),
            len(str(caption)))

def predict_ctr(img, caption):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    brightness, contrast, face_detected = extract_image_features(img_cv)
    sentiment_score, contains_cta, caption_len = extract_text_features(caption)
    features = pd.DataFrame([[sentiment_score, contains_cta, caption_len, brightness, contrast, face_detected]],
                            columns=["sentiment_score", "contains_cta", "caption_length", "brightness", "contrast", "face_detected"])
    prediction = model.predict(features)[0]
    shap_values = explainer.shap_values(features)
    return prediction, shap_values, features

st.title("AI-Driven Ad Creative Optimization DSS")

mode = st.radio("Select Mode", ("🎯 Single Image Prediction", "🆚 A/B Testing"))

if mode == "🎯 Single Image Prediction":
    uploaded_image = st.file_uploader("Upload Ad Image", type=["jpg", "jpeg", "png"])
    caption = st.text_area("Enter Caption")
    platform = st.selectbox("Choose Platform", ["Facebook", "Instagram"])
    if uploaded_image and caption:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Ad", use_container_width=True)
        pred, shap_vals, features = predict_ctr(img, caption)
        label = le.inverse_transform([pred])[0]
        log_prediction({"creative":"Single", "caption":caption, "platform":platform}, label)
        st.subheader(f"📊 Predicted CTR Class: **{label}**")
        with st.expander("🔍 Explanation (SHAP)"):
            with plt.style.context('ggplot'):
                fig, ax = plt.subplots()
                shap.summary_plot(shap_vals, features, plot_type="bar", show=False)
                st.pyplot(fig)
        st.subheader("📝 Recommendation")
        st.success("🚀 This ad is likely to perform well!" if label == "High" else "⚠️ This ad might need optimization.")

elif mode == "🆚 A/B Testing":
    st.header("Compare Two Creatives")
    col1, col2 = st.columns(2)
    with col1:
        image_a = st.file_uploader("Upload Creative A", type=["jpg", "jpeg", "png"], key="A")
        caption_a = st.text_area("Caption for A", key="A_cap")
    with col2:
        image_b = st.file_uploader("Upload Creative B", type=["jpg", "jpeg", "png"], key="B")
        caption_b = st.text_area("Caption for B", key="B_cap")
    platform = st.selectbox("Choose Platform", ["Facebook", "Instagram"])
    if image_a and caption_a and image_b and caption_b:
        if st.button("Predict CTR for A/B Test"):
            img_a = Image.open(image_a).convert("RGB")
            img_b = Image.open(image_b).convert("RGB")
            pred_a, shap_a, features_a = predict_ctr(img_a, caption_a)
            pred_b, shap_b, features_b = predict_ctr(img_b, caption_b)
            label_a, label_b = le.inverse_transform([pred_a])[0], le.inverse_transform([pred_b])[0]
            log_prediction({"creative":"A", "caption":caption_a, "platform":platform}, label_a)
            log_prediction({"creative":"B", "caption":caption_b, "platform":platform}, label_b)
            st.subheader("🔍 Prediction Results")
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**Creative A CTR:** {label_a}")
                with st.expander("🔍 SHAP for A"):
                    with plt.style.context('ggplot'):
                        figA, axA = plt.subplots()
                        shap.summary_plot(shap_a, features_a, plot_type="bar", show=False)
                        st.pyplot(figA)
            with colB:
                st.markdown(f"**Creative B CTR:** {label_b}")
                with st.expander("🔍 SHAP for B"):
                    with plt.style.context('ggplot'):
                        figB, axB = plt.subplots()
                        shap.summary_plot(shap_b, features_b, plot_type="bar", show=False)
                        st.pyplot(figB)
            st.subheader("📈 Recommended Creative:")
            if pred_a > pred_b:
                st.success("🎉 **Creative A** is recommended for better CTR!")
            elif pred_b > pred_a:
                st.success("🎉 **Creative B** is recommended for better CTR!")
            else:
                st.info("Both creatives have the same predicted CTR!")

