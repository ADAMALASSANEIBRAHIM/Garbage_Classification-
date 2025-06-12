
import streamlit as st
import numpy as np
import pandas as pd
import os
import cv2
from tensorflow.keras.models import load_model

# Optionnel : téléchargement auto du modèle si absent
try:
    import gdown
except ImportError:
    gdown = None

# --- Paramètres essentiels ---


MODEL_PATH = os.path.join("Models", "vgg16_garbage_classifier.h5")
CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
LOG_FILE = os.path.join(os.path.dirname(__file__), "prediction_logs.csv")
STYLE_PATH = os.path.join(os.path.dirname(__file__), "static", "style.css")

# Lien Google Drive du modèle (format gdown)
GDRIVE_URL = "https://drive.google.com/uc?id=1WIo8vdVmIoAm0VncBfbk-xMWkTLuB4rE"

# --- Chargement du modèle (avec cache et téléchargement auto) ---
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        if gdown is None:
            raise ImportError("Le module gdown n'est pas installé. Ajoutez 'gdown' à requirements.txt pour téléchargement auto du modèle.")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with st.spinner("Téléchargement du modèle VGG16 depuis Google Drive..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
        st.success("Modèle téléchargé.")
    return load_model(MODEL_PATH)

# --- Injection du CSS custom ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- Prédiction sur image numpy ---
def predict_image(img_np, model):
    img_resized = cv2.resize(img_np, (128, 128))
    img_array = np.expand_dims(img_resized, axis=0)
    preds = model.predict(img_array)
    return CATEGORIES[np.argmax(preds)]

# --- Enregistrement dans le log CSV ---
def log_prediction(image_name, predicted_class, true_class=None):
    log_data = {
        "image_name": image_name,
        "predicted_class": predicted_class,
        "true_class": true_class,
        "timestamp": pd.Timestamp.now()
    }
    # S'assurer que le dossier existe
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([log_data])], ignore_index=True)
    else:
        df = pd.DataFrame([log_data])
    df.to_csv(LOG_FILE, index=False)

# --- Affichage des statistiques ---
def show_stats():
    st.title("Statistiques de classification")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.metric("Nombre total de prédictions", len(df))
        st.download_button("Télécharger l'historique (CSV)", data=df.to_csv(index=False), file_name="prediction_logs.csv", mime="text/csv")
        if "true_class" in df.columns and df["true_class"].notnull().any():
            df_valid = df[df["true_class"].notnull()]
            accuracy = (df_valid["predicted_class"] == df_valid["true_class"]).mean()
            st.metric("Taux de bon classement (accuracy)", f"{accuracy*100:.1f}%")
        st.subheader("Répartition des classes prédites")
        st.bar_chart(df["predicted_class"].value_counts())
    else:
        st.info("Aucune prédiction enregistrée pour le moment.")

# --- Affichage du CSS custom (à utiliser dans l'app principale) ---
def inject_style():
    if os.path.exists(STYLE_PATH):
        local_css(STYLE_PATH)
