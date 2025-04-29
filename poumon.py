import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

def clean_dataset(base_dir):
    """Supprime les images corrompues et fichiers inutiles"""
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    num_deleted = 0
    num_checked = 0

    for subset in ["train", "val", "test"]:
        subset_dir = os.path.join(base_dir, subset)
        if not os.path.exists(subset_dir):
            print(f"Directory {subset_dir} not found, skipping.")
            continue
        for class_dir in os.listdir(subset_dir):
            class_path = os.path.join(subset_dir, class_dir)
            if not os.path.isdir(class_path):
                continue

            for fname in os.listdir(class_path):
                fpath = os.path.join(class_path, fname)
                num_checked += 1

                if not fname.lower().endswith(valid_extensions):
                    os.remove(fpath)
                    num_deleted += 1
                    print(f"Deleted non-image file: {fpath}")
                    continue

                try:
                    img = Image.open(fpath)
                    img.verify()
                except (IOError, SyntaxError):
                    os.remove(fpath)
                    num_deleted += 1
                    print(f"Deleted corrupted image: {fpath}")

    print(f"✅ Cleaned dataset: {num_deleted} files deleted over {num_checked} checked.")

# Clean dataset
dataset_path = r"model/chest_xray"
clean_dataset(dataset_path)

def load_data(data_dir, subset):
    """Load dataset from specific subset directory"""
    return image_dataset_from_directory(
        os.path.join(data_dir, subset),
        image_size=(224, 224),
        batch_size=32,
        label_mode='binary',
        shuffle=subset != "test"
    )

def create_model():
    """Create EfficientNetB0-based model with data augmentation"""
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
    ])

    base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True

    model = models.Sequential([
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Load datasets
data_dir = r"model/chest_xray"
train_dataset = load_data(data_dir, "train")
val_dataset = load_data(data_dir, "val")
test_dataset = load_data(data_dir, "test")

"""
data_dir = r"model/chest_xray/train"
train_dataset = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary'
)
val_dataset = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary'
)
test_dataset = load_data(r"model/chest_xray", "test")
"""

y_train = np.concatenate([y.numpy() for _, y in train_dataset], axis=0).flatten()
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# Create model
model = create_model()

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
]

# Train model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# Save model
model.save("model/chest_xray_model2.h5")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predict on test set for confusion matrix and classification report
y_true = np.concatenate([y.numpy() for _, y in test_dataset], axis=0).flatten()
y_pred = model.predict(test_dataset)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()

# Confusion matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_true, y_pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_labels, target_names=["Normal", "Pneumonia"]))

# Function to predict on a single image
def predict_image(img_path):
    """Predict on a single image"""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)[0][0]
        label = "Pneumonia" if prediction > 0.5 else "Normal"
        print(f"Prediction: {label} (Probability: {prediction:.2f})")

        plt.imshow(img)
        plt.title(f"Prediction: {label} (Probability: {prediction:.2f})")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Error predicting image {img_path}: {str(e)}")

