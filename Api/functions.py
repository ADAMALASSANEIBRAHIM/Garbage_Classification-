
import os
from tensorflow.keras.models import Model, load_model
import cv2 # lecture et traitement d'images
import numpy as np
import glob # pour la recherche de fichiers
# Ajout pour téléchargement du modèle depuis Google Drive
try:
    import gdown
except ImportError:
    gdown = None



# Enregistre une image décodée depuis du base64 dans un dossier donné
def save_image_from_base64(decoded_image, output_folder, output_filename):
    """
    Sauvegarde une image décodée depuis du base64 dans un dossier donné.
    Crée le dossier s'il n'existe pas.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, output_filename)
    with open(output_path, 'wb') as output_file:
        output_file.write(decoded_image)
    

# resize_images prend une liste d'images 
# et les redimensionne à une taille cible
def resize_images(images, target_size=(128, 128)):
    """Resize images to the target size using cv2."""
    resized_images = []
    for img in images:
        img_resized = cv2.resize(img, target_size)
        resized_images.append(img_resized)
    return np.array(resized_images)


# Nouvelle fonction avec nom plus intuitif et commentaires
def predict_garbage_image_class():
    """
    Charge le modèle et prédit la classe de l'image sauvegardée.
    Retourne le nom de la classe prédite.
    """
    # Chemin vers le modèle sauvegardé
    project_dir = os.path.dirname(os.getcwd())
    parent_dir = os.path.join(project_dir, 'API')
    model_dir = os.path.join(project_dir, 'models')
    model_path = os.path.join(model_dir, 'vgg16_garbage_classifier.h5')
    image_path = os.path.join(parent_dir, 'garbageImage', 'garbagePhoto.png')

    # Lien Google Drive (format gdown)
    # Lien Google Drive corrigé (format gdown)
    GDRIVE_URL = "https://drive.google.com/uc?id=1WIo8vdVmIoAm0VncBfbk-xMWkTLuB4rE"

    # Téléchargement automatique du modèle si absent
    if not os.path.exists(model_path):
        if gdown is None:
            raise ImportError("Le module gdown n'est pas installé. Ajoutez 'gdown' à requirements.txt.")
        os.makedirs(model_dir, exist_ok=True)
        print("Téléchargement du modèle VGG16 depuis Google Drive...")
        gdown.download(GDRIVE_URL, model_path, quiet=False)
        print("Modèle téléchargé.")

    img_array = cv2.imread(image_path)
    image = resize_images([img_array])
    # Charger le modèle
    model = load_model(model_path)
    predictions = model.predict(image)
    category = np.argmax(predictions, axis=1)
    categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    predicted_category = categories[category[0]]
    return predicted_category
