# Application de Détection Automatique de la Pneumonie à partir de Radiographies Pulmonaires avec Deep Learning



## 📌 Table des Matières
1. [Idée et objectifs du Projet](#-idée-et-objectifs-du-projet)
2. [Dataset](#-dataset)
3. [Prétraitement](#-prétraitement)
4. [Modèle](#-modèle)
5. [Résultats](#-résultats)
6. [Déploiement](#-déploiement)

## 🎯 Idée et Objectifs du Projet

La pneumonie est une infection respiratoire grave nécessitant un diagnostic rapide et précis. 
Le projet vise à automatiser la détection de la pneumonie à partir d'images de radiographies pulmonaires, afin d'assister les professionnels de santé et d'accélérer le processus de diagnostic.

### Objectifs :

- **Nettoyer un dataset de radiographies thoraciques** pour garantir la qualité des données.
- **Construire un modèle de classification binaire (Normal vs Pneumonia)** à l'aide de la deep learning.
- **Optimiser l'entraînement** avec des stratégies de fine-tuning, early stopping et réduction du taux d'apprentissage.
- **Évaluer les performances du modèle** avec des métriques claires : accuracy, confusion matrix, classification report.
- **Développer une fonction de prédiction d'images individuelles** pour tester facilement de nouvelles radiographies.


## 📊 Dataset

**Source** : Chest X-Ray Images (Pneumonia) — National Institutes of Health (NIH) / Kaggle

| Caractéristique          | Détails                            |
|--------------------------|------------------------------------|
| Images d'entraînement	   | 1,130 IRM annotées                 |
| Images de validation     | 120 IRM                            |
| Format                   | .jpeg, .png, .bmp                  |
| Résolution               | Variable (redimensionnée à 224×224)|
| Annotations              | Binaires (0=normal, 1=Pneumonia)   |

**📊 Compréhension des Données** :
 Structure des données
Images IRM au format .jpeg, .png, .bmp
Chaque IRM est en niveaux de gris (parfois RGB, converties si nécessaire).
Résolution originale :  224×224 pixels
 **🔢 Annotations binaires**
Anomalie générale : 0 = normal, 1 = Pneumonia
**📑 Métadonnées**
Fichiers : train-abnormal.csv, train-acl.csv, train-meniscus.csv


## 🛠 Préparation des Données

**Nettoyage :**
Suppression des fichiers corrompus et formats incorrects.
**Prétraitement :**
Redimensionnement des images en 224×224 pixels.
Normalisation des pixels dans l'intervalle [0,1].
**Chargement :**
Utilisation de image_dataset_from_directory de TensorFlow pour générer les ensembles de données avec étiquetage automatique.


## 🧠 Modélisation

| Couche                  | Détails                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| Entrée                  | Image (224 × 224 × 3)                                                   |
| EfficientNetB0	        | Base pré-entraînée sur ImageNet, sans la couche top (include_top=False) | 
| GlobalAveragePooling2D	| Réduction de dimension spatiale                                         |
| MaxPooling2D            | Taille 2×2                                                              |                                                                            
| Dense                   | 256 puis 128 neurones                                                   |
| Dropout                 | Taux de dropout : 0.5                                                   |

# 5. 🧪 Résultats
**📊 Métriques principales sur le Test Set :**
Exactitude (Accuracy) : ~92.3 %
Précision : ~91.8 %
Rappel (Recall) : ~92.7 %
Score F1 : ~92.2 %

## 🚀 Déploiement
Application avec Streamlit
L’application déployée permet :
 - Upload d'une radiographie thoracique (format JPG, JPEG, PNG)
 - Analyse automatique via le modèle EfficientNetB0 entraîné :
    * Détection d'une radiographie Normale ou atteinte de Pneumonie
 - Affichage de la prédiction :
    * Prédiction de la classe ("Normal" ou "Pneumonia")
 - Affichage de la probabilité associée
 -  Visualisation de l’image dans l’interface

