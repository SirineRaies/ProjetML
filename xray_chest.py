import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def load_data(data_dir, subset):
    """Load image dataset from a specified subset directory"""
    return image_dataset_from_directory(
        os.path.join(data_dir, subset),
        image_size=(224, 224),
        batch_size=32,
        label_mode='binary'
    )

def create_model():
    """Create EfficientNetB0-based binary classification model"""
    base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True

    model = models.Sequential([
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

def train_model():
    """Train model and save .h5 file in /model"""
    data_dir = "./model/chest_xray"

    train_dataset = load_data(data_dir, "train")
    val_dataset = load_data(data_dir, "val")
    test_dataset = load_data(data_dir, "test")

    model = create_model()

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=callbacks
    )

    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"✅ Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

    # Save the model
    model_path = "./model/chest_xray_model.h5"
    model.save(model_path)
    print(f"✅ Model saved at: {model_path}")

    return history

def main():
    train_model()

if __name__ == "__main__":
    main()
