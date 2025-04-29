# Application de Détection Automatique de la Pneumonie à partir de Radiographies Pulmonaires avec Deep Learning

---

## 📌 Table des Matières
1. [Idée et Objectifs du Projet](#-idée-et-objectifs-du-projet)
2. [Dataset](#-dataset)
3. [Prétraitement](#-prétraitement)
4. [Modèle](#-modèle)
5. [Résultats](#-résultats)
6. [Déploiement](#-déploiement)

---

## 🎯 Idée et Objectifs du Projet

La pneumonie est une infection respiratoire grave nécessitant un diagnostic rapide et précis.  
Ce projet vise à automatiser la détection de la pneumonie à partir de radiographies pulmonaires, afin d'assister les professionnels de santé et d'accélérer le processus de diagnostic.

### Objectifs :

- **Nettoyer un dataset de radiographies thoraciques** pour garantir la qualité des données.
- **Construire un modèle de classification binaire** (Normal vs Pneumonia) à l'aide du deep learning.
- **Optimiser l'entraînement** avec des stratégies de fine-tuning, early stopping et réduction du taux d'apprentissage.
- **Évaluer les performances du modèle** avec des métriques claires : accuracy, confusion matrix, classification report.
- **Développer une fonction de prédiction d'images individuelles** pour tester facilement de nouvelles radiographies.

---

## 📊 Dataset

**Source** : *Chest X-Ray Images (Pneumonia) — National Institutes of Health (NIH) / Kaggle*

| Caractéristique          | Détails                            |
|---------------------------|------------------------------------|
| Images d'entraînement     | 3,000 images                      |
| Images de validation      | 856 images                        |
| Images de test            | 624 images                        |
| Format                    | JPEG, PNG ,JPG ,BMP               |
| Résolution                | Variable (redimensionnée à 224×224)|
| Annotations               | Binaire (0 = Normal, 1 = Pneumonia)|

### 📂 Structure des Données

- Images au format `.jpeg`, `.png`,`.jpg`, `.bmp`
- Certaines images en niveaux de gris, converties en RGB si nécessaire
- Résolution standardisée : **224×224 pixels**
- Labels : 
  - **0 = Normal**
  - **1 = Pneumonia**

---

## 🛠 Prétraitement

**Nettoyage :**
- Suppression des fichiers corrompus et formats incorrects.

**Prétraitement :**
- Redimensionnement des images à **224×224 pixels**.
- Normalisation des pixels dans l'intervalle [0,1].

**Chargement :**
- Utilisation de `image_dataset_from_directory` (TensorFlow) pour charger et étiqueter automatiquement les images.

---

## 🧠 Modèle

| Couche                  | Détails                                                                 |
|--------------------------|-------------------------------------------------------------------------|
| Entrée                   | Image (224 × 224 × 3)                                                   |
| EfficientNetB0           | Base pré-entraînée sur ImageNet (`include_top=False`)                   |
| GlobalAveragePooling2D   | Réduction de dimension spatiale                                         |
| Dense                    | 256 neurones, activation ReLU                                           |
| Dropout                  | Taux de dropout : 0.5                                                   |
| Dense                    | 128 neurones, activation ReLU                                           |
| Dense (Sortie)            | 1 neurone, activation Sigmoïde (classification binaire)                |

---

## 🧪 Résultats

**📊 Métriques principales sur le Test Set :**

- **Exactitude (Accuracy)** : ~92.3 %
- **Précision** : ~91.8 %
- **Rappel (Recall)** : ~92.7 %
- **Score F1** : ~92.2 %

---

## 🚀 Déploiement

**Application avec Streamlit**

L’application déployée permet :

- 📤 **Upload d'une radiographie thoracique** (formats JPG, JPEG, PNG)
- 🤖 **Analyse automatique** via le modèle EfficientNetB0 :
  - Détection d'une radiographie **Normale** ou **atteinte de Pneumonie**
- 📊 **Affichage des résultats** :
  - Prédiction de la classe ("Normal" ou "Pneumonia")
  - Affichage de la **probabilité associée**
- 🖼️ **Visualisation de l’image** directement dans l’interface utilisateur.

---

## ✅ Remarque

Le projet est déployable localement avec :

```bash
streamlit run app.py
