# Plateforme de Classification d'Images de Déchets

Plateforme professionnelle, pédagogique et autonome pour la classification d'images de déchets.

## Fonctionnalités principales
- **Classification d'images** via une interface web ergonomique
- **API Flask** robuste pour la prédiction (mode legacy)
- **Dashboard Streamlit** interactif (upload, stats, export CSV)
- **Application Streamlit autonome** prête pour le cloud (dossier `deployment_1`)
- **Persistance des résultats** et visualisation dynamique
- **Modèle VGG16 décoré** (fine-tuning sur dataset déchets)


## Structure du projet
```
API/             # API Flask pour la prédiction (optionnel)
Dashboard/       # Dashboard Streamlit (legacy)
deployment_1/    # Nouvelle application Streamlit tout-en-un (recommandé)
Models/          # Modèles entraînés (VGG16, etc.)
requirements.txt, .gitignore, README.md
```

## Installation rapide (Application autonome recommandée)
1. **Cloner le repo**
   ```bash
   git clone https://github.com/ADAMALASSANEIBRAHIM/Garbage_Classification-.git
   cd Garbage_Classification
   ```
2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Sous Git Bash
   # ou
   .\venv\Scripts\Activate.ps1  # Sous PowerShell
   ```
3. **Installer les dépendances**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Installer l'API**
```bash
   cd Api
   python app.py
```
4. **Lancer l'application Streamlit autonome**
   ```bash
      cd Dashboard
      streamlit run app.py
   ```

## Pour utiliser l'API Flask ou l'ancien dashboard
Voir les dossiers `API/` et `Dashboard/` pour les instructions legacy.

## Déploiement
- **Local** : voir ci-dessus
- **Cloud** : Docker, Streamlit Cloud, Render, Heroku, Azure, etc. (voir section Déploiement dans le code ou demander un exemple de Dockerfile)

## Modèle utilisé
Le modèle principal est **VGG16 décoré** 
Ce modèle est choisi suite à une validation croisée entre 3 modèle, à savoir: le Resnet50, le CNN simple et bien évidemment le VGG16.
## Contact
adamalassaneibrahim@gmail.com 
