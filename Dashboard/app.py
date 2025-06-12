

# --- Dashboard Streamlit avancé pour classification d'image de déchets (tout-en-un, sans API) ---
import streamlit as st
import pandas as pd
import os
import numpy as np
import cv2
from deployment_1.garb_class import get_model, predict_image, log_prediction, inject_style, show_stats


st.set_page_config(page_title="Classification de déchets", layout="wide")
inject_style()


# Sidebar navigation

# Sidebar navigation inspirée d'Amazon/Azure, sans logo de marque
st.sidebar.markdown("""
<div style='font-size:1.3em; font-weight:bold; color:#0078d4; margin-bottom:0.5em; white-space:normal; word-break:break-word;'>Déchets Classifier</div>
<div style='font-size:1em; color:#232f3e; margin-bottom:1em; white-space:normal; word-break:break-word;'>Plateforme professionnelle de classification</div>
""", unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["Classification", "Statistiques", "Aide"])

# Fichier CSV pour enregistrer l'historique des prédictions
LOG_FILE = "prediction_logs.csv"

if page == "Classification":
    st.title("Classification d'image de déchets")
    st.write("Uploadez une image de déchet, le modèle prédit la classe localement.")

    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    true_label = st.text_input("(Optionnel) Classe attendue (pour statistiques)")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Image chargée", use_column_width=True)
        img_bytes = uploaded_file.read()
        # Conversion en image numpy
        file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_np is None:
            st.markdown(f'<div class="custom-toast error">Erreur de lecture de l\'image. Format non supporté.</div>', unsafe_allow_html=True)
        else:
            if st.button("Classer l'image"):
                try:
                    model = get_model()
                    predicted = predict_image(img_np, model)
                    st.markdown(f'<span class="status-badge success">Succès</span>', unsafe_allow_html=True)
                    st.success(f"Classe prédite : {predicted}")
                    log_prediction(uploaded_file.name, predicted, true_label)
                except Exception as e:
                    st.markdown(f'<div class="custom-toast error">Erreur lors de la prédiction : {e}</div>', unsafe_allow_html=True)

elif page == "Statistiques":
    show_stats()


elif page == "Aide":
    st.title("Aide et informations")
    st.markdown("""
    <ul style="list-style:none; padding-left:0;">
      <li>🖼️ <b>Classification</b> : Uploadez une image, cliquez sur <b>Classer l'image</b> pour obtenir la prédiction.</li>
      <li>📊 <b>Statistiques</b> : Consultez l'historique des prédictions et les performances du modèle.</li>
      <li>⬇️ <b>Export</b> : Téléchargez l'historique des prédictions au format CSV.</li>
    </ul>
    <hr style="margin:1em 0;"/>
    <div style="display:flex;align-items:flex-start;gap:1.2em;flex-wrap:wrap;">
      <div style="font-size:2.2em;line-height:1;">🤖</div>
      <div style="min-width:220px;">
        <b>Description du modèle de classification</b><br>
        Le système utilise un modèle d’intelligence artificielle appelé <b>VGG16</b>, une architecture de réseau de neurones profonde reconnue pour ses performances en reconnaissance d’images.<br>
        <span style="color:#0078d4;font-weight:500;">VGG16 décoré</span> a été initialement entraîné sur des millions d’images pour apprendre à reconnaître des objets variés. Dans notre cas, ce modèle a été décoré (adapté et affiné) spécifiquement pour la classification des déchets, en réutilisant ses connaissances générales et en les spécialisant sur notre jeu de données.
      </div>
    </div>
    <div style="display:flex;align-items:flex-start;gap:1.2em;flex-wrap:wrap;margin-top:1.2em;">
      <div style="font-size:2em;">🔎</div>
      <div style="min-width:220px;">
        <b>Comment ça marche ?</b><br>
        Lorsqu’une image de déchet est envoyée, le modèle analyse ses caractéristiques visuelles (formes, couleurs, textures) et compare ce qu’il “voit” à ce qu’il a appris lors de son entraînement. Il attribue ensuite la catégorie la plus probable (plastique, papier, métal, etc.).
      </div>
    </div>
    <div style="display:flex;align-items:flex-start;gap:1.2em;flex-wrap:wrap;margin-top:1.2em;">
      <div style="font-size:2em;">❓</div>
      <div style="min-width:220px;">
        <b>Pourquoi le taux de bon classement peut-il être à 0,0 % ?</b><br>
        <ul style="margin:0 0 0 1.2em;">
          <li>Si la colonne “classe attendue” (vérité terrain) n’est pas renseignée lors des prédictions, le calcul de l’accuracy n’est pas possible.</li>
          <li>Si les classes attendues saisies ne correspondent pas exactement aux classes prédites (erreur de frappe, casse, etc.), l’accuracy sera faussement basse.</li>
          <li>Vérifiez que vous indiquez la classe attendue lors de chaque prédiction pour obtenir un taux de bon classement représentatif.</li>
        </ul>
      </div>
    </div>
    <div style="display:flex;align-items:flex-start;gap:1.2em;flex-wrap:wrap;margin-top:1.2em;">
      <div style="font-size:2em;">💡</div>
      <div style="min-width:220px;">
        <b>Résumé professionnel :</b><br>
        <div style="background:#f8fafd;border-left:5px solid #0078d4;padding:1em 1.2em;margin-top:0.5em;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.07);color:#232f3e;">
          <i>Le modèle VGG16 décoré offre une classification robuste et rapide des images de déchets, en s’appuyant sur une expertise visuelle acquise sur de larges bases d’images, puis adaptée à la problématique du tri sélectif. Son usage permet d’automatiser et de fiabiliser la reconnaissance des déchets à partir de simples photos.</i>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="status-badge waiting">Plateforme professionnelle</div>', unsafe_allow_html=True)

# Footer professionnel
st.markdown('<div class="custom-footer">© 2025 Plateforme Classification Déchets</div>', unsafe_allow_html=True)
