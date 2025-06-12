
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import base64  # Pour décoder les images base64
# Importation des fonctions nécessaires. Ces 
# fonctions sont définies dans le fichier functions.py
from functions import save_image_from_base64, predict_garbage_image_class

# Création de l'application Flask et
# configuration CORS
app = Flask(__name__)
CORS(app)

# Configuration de l'application Flask
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/classifyGarbage", methods=['POST'])
def classify_garbage():
    try:
        data = request.get_json()
        if not data or 'garbageImage' not in data:
            return jsonify({"error": "Aucune image reçue (clé 'garbageImage' manquante)"}), 400
        garbage_image_encoded = data.get('garbageImage')
        try:
            garbage_image_decoded = base64.b64decode(garbage_image_encoded)
        except Exception as e:
            return jsonify({"error": f"Erreur lors du décodage base64 : {str(e)}"}), 400
        # Dossier et nom cohérents avec functions.py
        image_folder = "garbageImage"
        image_filename = "garbagePhoto.png"
        save_image_from_base64(garbage_image_decoded, image_folder, image_filename)
        predicted_class = predict_garbage_image_class()
        return jsonify({"classe": predicted_class})
    except Exception as e:
        return jsonify({"error": f"Erreur serveur : {str(e)}"}), 500

if __name__ == "__main__":
    app.run()
