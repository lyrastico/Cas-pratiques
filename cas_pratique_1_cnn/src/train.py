import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from cas_pratique_1_cnn.src.utils import plot_training_curves


MODEL_PATH = "models/cnn_cifar10.keras"
RESULTS_PATH = "results/training_curves.png"


def load_data():
    """Charge et prépare le dataset CIFAR-10."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalisation : les pixels passent de [0, 255] à [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Encodage one-hot des labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def build_model():
    """Construit un CNN simple pour classifier les images CIFAR-10."""
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    x_train, y_train, x_test, y_test = load_data()
    model = build_model()

    print("Résumé du modèle :")
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("\nDébut de l'entraînement...")
    history = model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
    )

    print("\nÉvaluation sur les données de test :")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_accuracy:.4f}")

    model.save(MODEL_PATH)
    plot_training_curves(history, RESULTS_PATH)

    print(f"\nModèle sauvegardé dans : {MODEL_PATH}")
    print(f"Courbes sauvegardées dans : {RESULTS_PATH}")

if __name__ == "__main__":
    main()
