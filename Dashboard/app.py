
# --- Dashboard Streamlit avancé pour classification d'image de déchets via l'API Flask ---
import streamlit as st
import requests
import base64
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Classification de déchets", layout="wide")


# Injection du CSS custom
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css(os.path.join("static", "style.css"))


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
    st.write("Uploadez une image de déchet, le modèle prédit la classe via l'API Flask.")

    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    true_label = st.text_input("(Optionnel) Classe attendue (pour statistiques)")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Image chargée", use_column_width=True)
        img_bytes = uploaded_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        api_url = "http://127.0.0.1:5000/classifyGarbage"
        data = {"garbageImage": img_b64}
        if st.button("Classer l'image"):
            try:
                response = requests.post(api_url, json=data)
                if response.status_code == 200:
                    result = response.json()
                    predicted = result.get('classe', 'Inconnue')
                    # Badge de statut
                    st.markdown(f'<span class="status-badge success">Succès</span>', unsafe_allow_html=True)
                    # Affichage direct de la classe prédite
                    st.success(f"Classe prédite : {predicted}")
                    # Enregistrement dans le log CSV
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
                else:
                    st.markdown(f'<div class="custom-toast error">Erreur API : {response.status_code} - {response.text}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="custom-toast error">Erreur de connexion à l\'API : {e}</div>', unsafe_allow_html=True)

elif page == "Statistiques":
    st.title("Statistiques de classification")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.metric("Nombre total de prédictions", len(df))
        # Export CSV
        st.download_button("Télécharger l'historique (CSV)", data=df.to_csv(index=False), file_name="prediction_logs.csv", mime="text/csv")
        if "true_class" in df.columns and df["true_class"].notnull().any():
            df_valid = df[df["true_class"].notnull()]
            accuracy = (df_valid["predicted_class"] == df_valid["true_class"]).mean()
            st.metric("Taux de bon classement (accuracy)", f"{accuracy*100:.1f}%")
        st.subheader("Répartition des classes prédites")
        st.bar_chart(df["predicted_class"].value_counts())
        # Toast info
        st.markdown('<div class="custom-toast info">Statistiques à jour. Données exportables.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="custom-toast info">Aucune prédiction enregistrée pour le moment.</div>', unsafe_allow_html=True)


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
