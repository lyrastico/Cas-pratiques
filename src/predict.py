import sys

import numpy as np
import tensorflow as tf
from PIL import Image

from utils import CLASS_NAMES


MODEL_PATH = "models/cnn_cifar10.keras"


def prepare_image(image_path):
    """Charge une image et la prépare au format attendu par le CNN."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((32, 32))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def main():
    if len(sys.argv) < 2:
        print("Utilisation : python src/predict.py chemin/vers/image.jpg")
        return

    image_path = sys.argv[1]
    model = tf.keras.models.load_model(MODEL_PATH)
    image_array = prepare_image(image_path)

    prediction = model.predict(image_array)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    print(f"Classe prédite : {CLASS_NAMES[predicted_index]}")
    print(f"Confiance      : {confidence:.2%}")


if __name__ == "__main__":
    main()
