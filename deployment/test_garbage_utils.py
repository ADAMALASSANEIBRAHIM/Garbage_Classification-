import streamlit as st
import numpy as np
import cv2
from garb_class import get_model, predict_image, show_stats, inject_style, log_prediction

inject_style()
st.title("Test rapide du module garb_class.py")

model = get_model()

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", caption="Image chargée", use_column_width=True)
    if st.button("Prédire"):
        predicted = predict_image(img, model)
        st.success(f"Catégorie prédite : {predicted}")
        log_prediction(uploaded_file.name, predicted)
        show_stats()
