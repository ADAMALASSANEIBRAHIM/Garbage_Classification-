# --- Dashboard Streamlit avancé pour classification d'image de déchets (sans dépendance API) ---
import streamlit as st
import base64
import pandas as pd
import os
import pathlib
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

st.set_page_config(page_title="Classification de déchets", layout="wide")

# Chargement du modèle entraîné (personnel)
MODEL_PATH = "./model/model_dechets.h5"
model = load_model(MODEL_PATH)
CLASSES = ["Plastique", "Papier", "Verre", "Métal", "Organique"]

# CSS personnalisé
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path(__file__).parent / "static" / "style.css"
local_css(str(css_path))

# Sidebar
st.sidebar.markdown("""
<div style='font-size:1.3em; font-weight:bold; color:#0078d4; margin-bottom:0.5em;'>Déchets Classifier</div>
<div style='font-size:1em; color:#232f3e; margin-bottom:1em;'>Plateforme professionnelle de classification</div>
""", unsafe_allow_html=True)

with st.sidebar:
    page = option_menu(
        None,
        ["Classification", "Statistiques", "Aide"],
        icons=["image", "bar-chart", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f3f3f3"},
            "icon": {"color": "#388e3c", "font-size": "1.3em"},
            "nav-link": {
                "font-size": "1.1em",
                "text-align": "left",
                "margin": "0.2em 0",
                "border-radius": "8px",
                "color": "#232f3e",
                "background-color": "#e8f5e9",
            },
            "nav-link-selected": {
                "background-color": "#43a047",
                "color": "white",
                "font-weight": "bold",
            },
        }
    )

LOG_FILE = "prediction_logs.csv"

def predict_image_class(image: Image.Image):
    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    predicted_class = CLASSES[class_index]
    return predicted_class

if page == "Classification":
    st.title("Classification d'image de déchets")
    st.write("Uploadez une image de déchet, le modèle prédit la classe sans dépendre d'une API.")

    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    true_label = st.text_input("(Optionnel) Classe attendue (pour statistiques)")

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Image chargée", use_column_width=True)

        if st.button("Classer l'image"):
            try:
                predicted = predict_image_class(img)
                st.markdown(f'<span class="status-badge success">Succès</span>', unsafe_allow_html=True)
                st.success(f"Classe prédite : {predicted}")
                log_data = {
                    "image_name": uploaded_file.name,
                    "predicted_class": predicted,
                    "true_class": true_label,
                    "timestamp": pd.Timestamp.now()
                }
                if os.path.exists(LOG_FILE):
                    df = pd.read_csv(LOG_FILE)
                    df = pd.concat([df, pd.DataFrame([log_data])], ignore_index=True)
                else:
                    df = pd.DataFrame([log_data])
                df.to_csv(LOG_FILE, index=False)
            except Exception as e:
                st.markdown(f'<div class="custom-toast error">Erreur lors de la classification : {e}</div>', unsafe_allow_html=True)

elif page == "Statistiques":
    st.title("Statistiques de classification")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.metric("Nombre total de prédictions", len(df))
        st.download_button("Télécharger l'historique (CSV)", data=df.to_csv(index=False), file_name="prediction_logs.csv", mime="text/csv")
        st.subheader("Répartition des classes prédites")
        class_counts = df["predicted_class"].value_counts()
        fig, ax = plt.subplots()
        class_counts.plot(kind="bar", color="#43a047", ax=ax)
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Nombre")
        ax.set_title("Répartition des classes prédites")
        st.pyplot(fig)

        if df["true_class"].notnull().any():
            st.subheader("Matrice de confusion et rapport de classification")
            filtered_df = df[df["true_class"].notnull()]
            y_true = filtered_df["true_class"].values
            y_pred = filtered_df["predicted_class"].values
            cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues", ax=ax2)
            ax2.set_xlabel("Classe prédite")
            ax2.set_ylabel("Classe réelle")
            ax2.set_title("Matrice de confusion")
            st.pyplot(fig2)

            report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True)
            st.subheader("Rapport de classification")
            st.dataframe(pd.DataFrame(report).transpose().round(2))

        st.markdown('<div class="custom-toast info">Statistiques à jour. Données exportables.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="custom-toast info">Aucune prédiction enregistrée pour le moment.</div>', unsafe_allow_html=True)

elif page == "Aide":
    st.title("Aide et informations")
    st.markdown("""
    <ul style="list-style:none; padding-left:0;">
      <li>🖼️ <b>Classification</b> : Uploadez une image, cliquez sur <b>Classer l'image</b> pour obtenir la prédiction.</li>
      <li>📊 <b>Statistiques</b> : Consultez l'historique des prédictions.</li>
      <li>⬇️ <b>Export</b> : Téléchargez l'historique des prédictions au format CSV.</li>
    </ul>
    <hr style="margin:1em 0;"/>
    <div style="display:flex;align-items:flex-start;gap:1.2em;flex-wrap:wrap;">
      <div style="font-size:2.2em;line-height:1;">🤖</div>
      <div style="min-width:220px;">
        <b>Description du modèle de classification</b><br>
        Le système utilise un modèle d’intelligence artificielle <b>entraîné spécifiquement sur des images de déchets</b> (plastique, métal, etc.). Ce modèle a été construit à partir d’un jeu de données adapté au contexte local pour fournir une classification réaliste et pertinente.<br>
        <span style="color:#0078d4;font-weight:500;">Aucun service externe ni API n’est requis</span> : tout se fait en local.
      </div>
    </div>
    <div style="display:flex;align-items:flex-start;gap:1.2em;flex-wrap:wrap;margin-top:1.2em;">
      <div style="font-size:2em;">🔎</div>
      <div style="min-width:220px;">
        <b>Comment ça marche ?</b><br>
        Lorsqu’une image de déchet est envoyée, le modèle analyse ses caractéristiques visuelles (formes, couleurs, textures) et compare ce qu’il “voit” à ce qu’il a appris lors de son entraînement. Il attribue ensuite la catégorie la plus probable (plastique, papier, métal, etc.).
      </div>
    </div>
    <div style="display:flex;align-items:flex-start;gap:1.2em;flex-wrap:wrap;margin-top:1.2em;">
      <div style="font-size:2em;">💡</div>
      <div style="min-width:220px;">
        <b>Résumé professionnel :</b><br>
        <div style="background:#f8fafd;border-left:5px solid #0078d4;padding:1em 1.2em;margin-top:0.5em;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.07);color:#232f3e;">
          <i>Le modèle personnalisé de classification de déchets permet une reconnaissance rapide et fiable à partir d’images. Il est conçu pour fonctionner localement, sans connexion à une API externe, ce qui le rend parfaitement adapté aux déploiements autonomes.</i>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Pied de page
st.markdown('<div class="custom-footer">© 2025 Plateforme Classification Déchets</div>', unsafe_allow_html=True)
