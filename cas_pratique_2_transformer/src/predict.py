import re
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# On importe les classes custom pour que Keras puisse recharger le modèle.
from train import TokenAndPositionEmbedding, TransformerBlock, MAX_FEATURES, MAX_LEN


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "transformer_imdb.keras"


def clean_text(text):
    """Nettoyage simple du texte."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9']+", " ", text)
    return text.strip()


def text_to_sequence(text, word_index):
    """Convertit une phrase en séquence compatible avec le vocabulaire IMDB."""
    # Dans le dataset IMDB de Keras, les indices 0, 1 et 2 sont réservés.
    sequence = []
    for word in clean_text(text).split():
        index = word_index.get(word)
        if index is not None and index + 3 < MAX_FEATURES:
            sequence.append(index + 3)
        else:
            sequence.append(2)  # mot inconnu
    return pad_sequences([sequence], maxlen=MAX_LEN)


def main():
    if len(sys.argv) < 2:
        print('Utilisation : python src\\predict.py "This movie was amazing"')
        return

    text = " ".join(sys.argv[1:])

    if not MODEL_PATH.exists():
        print(f"Modèle introuvable : {MODEL_PATH}")
        print("Lance d'abord : python src\\train.py")
        return

    custom_objects = {
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock,
    }
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

    word_index = imdb.get_word_index()
    sequence = text_to_sequence(text, word_index)

    probability = float(model.predict(sequence, verbose=0)[0][0])
    label = "positif" if probability >= 0.5 else "négatif"

    print(f"Texte : {text}")
    print(f"Probabilité avis positif : {probability:.4f}")
    print(f"Prédiction : avis {label}")


if __name__ == "__main__":
    main()
