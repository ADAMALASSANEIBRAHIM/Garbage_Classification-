# Plateforme de Classification d'Images de Déchets

Plateforme professionnelle, pédagogique et autonome pour la classification d'images de déchets, avec API Flask, dashboard Streamlit moderne, et notebook d'analyse.

## Fonctionnalités principales
- **Classification d'images** via une interface web ergonomique
- **API Flask** robuste pour la prédiction
- **Dashboard Streamlit** interactif (upload, stats, export CSV)
- **Persistance des résultats** et visualisation dynamique
- **Modèle VGG16 décoré** (fine-tuning sur dataset déchets)
- **Design responsive, dark mode, notifications toast, badges, loader animé**

## Structure du projet
```
API/           # API Flask pour la prédiction
Dashboard/     # Dashboard Streamlit (frontend)
analyse.ipynb  # Notebook d'analyse pédagogique
models/        # Modèles entraînés (VGG16, etc.)
requirements.txt, .gitignore, README.md
```

## Installation rapide
1. **Cloner le repo**
   ```bash
   git clone https://github.com/ADAMALASSANEIBRAHIM/Garbage_Classification-.git
   cd Garbage_Classification
   ```
# Créer un environnement virtuel
python3.10 -m venv venv
source venv/Scripts/activate  # Sous Git Bash
# ou
.\venv\Scripts\Activate.ps1  # Sous PowerShell

2. **Installer les dépendances**
Il faut souvent mettre à jour le pip install.
   ```bash
   pip install -r requirements.txt
   ```
3. **Lancer l'API Flask** Faire en étant dans la racine du projet
   ```bash
   cd API
   python app.py
   ```
4. **Lancer le dashboard Streamlit**

   ```bash
   cd Dashboard
   streamlit run app.py
   ```

## Déploiement
- **Local** : voir ci-dessus
- **Cloud** : Docker, Streamlit Cloud, Render, Heroku, Azure, etc. (voir section Déploiement dans le code ou demander un exemple de Dockerfile)

## Modèle utilisé
Le modèle principal est **VGG16 décoré** 
Ce modèle est choisi suite à une validation croisée entre 3 modèle, à savoir: le Resnet50, le CNN simple et bien évidemment le VGG16.
## Contact
adamalassaneibrahim@gmail.com 
