import importlib
import subprocess
import sys

# Liste des bibliothèques à vérifier
required_libs = {
    "kaggle": "kaggle",
    "keras": "keras",
    "lazypredict": "lazypredict",
    "loguru": "loguru",
    "opencv_python": "opencv-python",
    "pandas": "pandas",
    "Pillow": "Pillow",
    "pytest": "pytest",
    "scikeras": "scikeras",
    "scikit_learn": "scikit-learn",
    "seaborn": "seaborn",
    "tensorflow": "tensorflow",
    "mlflow": "mlflow",
    "flask": "flask",
    "flask_restful": "flask-restful",
    "flask_cors": "flask-cors"
}

installed = []
missing = []

# Vérification de l'import
for import_name, pip_name in required_libs.items():
    try:
        # Spécial cas pour opencv
        if import_name == "opencv_python":
            importlib.import_module("cv2")
        elif import_name == "Pillow":
            importlib.import_module("PIL")
        elif import_name == "scikit_learn":
            importlib.import_module("sklearn")
        elif import_name == "flask_restful":
            importlib.import_module("flask_restful")
        elif import_name == "flask_cors":
            importlib.import_module("flask_cors")
        else:
            importlib.import_module(import_name)
        installed.append(pip_name)
    except ImportError:
        missing.append(pip_name)

# Résumé
print("✅ Bibliothèques installées :", installed)
print("❌ Bibliothèques manquantes :", missing)

# Installation si nécessaire
if missing:
    choice = input("\nSouhaitez-vous installer les bibliothèques manquantes ? (y/n): ").strip().lower()
    if choice == 'y':
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        print("\n✅ Installation terminée.")
    else:
        print("⏭️ Installation ignorée.")
else:
    print("\n🎉 Tout est déjà installé !")
