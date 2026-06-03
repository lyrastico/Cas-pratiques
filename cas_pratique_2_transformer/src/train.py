import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import plot_training_curves


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "transformer_imdb.keras"
RESULTS_PATH = BASE_DIR / "results" / "training_curves.png"

MAX_FEATURES = 20_000   # Nombre maximum de mots dans le vocabulaire
MAX_LEN = 200           # Longueur maximale d'un texte après padding
EMBED_DIM = 64          # Taille des vecteurs d'embedding
NUM_HEADS = 2           # Nombre de têtes d'attention
FF_DIM = 64             # Taille du réseau feed-forward interne


class TransformerBlock(layers.Layer):
    """Bloc Transformer simple : attention + feed-forward."""

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    """Embedding des mots + embedding de leur position dans la phrase."""

    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def load_data():
    """Charge et prépare le dataset IMDB."""
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

    x_train = pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = pad_sequences(x_test, maxlen=MAX_LEN)

    return x_train, y_train, x_test, y_test


def build_model():
    """Construit un modèle Transformer pour la classification binaire."""
    inputs = layers.Input(shape=(MAX_LEN,))
    x = TokenAndPositionEmbedding(MAX_LEN, MAX_FEATURES, EMBED_DIM)(inputs)
    x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    os.makedirs(BASE_DIR / "models", exist_ok=True)
    os.makedirs(BASE_DIR / "results", exist_ok=True)

    x_train, y_train, x_test, y_test = load_data()
    model = build_model()

    print("Résumé du modèle :")
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    print("\nDébut de l'entraînement...")
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
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
