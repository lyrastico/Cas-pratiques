# Cas pratique n°1 — Classification d’images avec un CNN

Projet de base pour commencer le TP sur VS Code.

## Objectif
Créer un modèle de Deep Learning capable de classer des images avec un réseau de neurones convolutif, ou CNN.

Par défaut, le projet utilise le dataset **CIFAR-10** intégré à TensorFlow/Keras. Il contient 10 classes : avion, voiture, oiseau, chat, cerf, chien, grenouille, cheval, bateau, camion.

## Structure du dossier

```text
cas_pratique_1_cnn/
├── data/                 # Données si tu ajoutes ton propre dataset
├── models/               # Modèles sauvegardés
├── notebooks/            # Notebook de test ou d'explication
├── results/              # Graphiques et résultats
├── src/
│   ├── train.py           # Entraînement du CNN
│   ├── predict.py         # Prédiction sur une image
│   └── utils.py           # Fonctions utiles
├── requirements.txt       # Librairies à installer
├── .gitignore
└── README.md
```

## Installation

Dans VS Code, ouvre le dossier puis lance dans le terminal :

```bash
python -m venv .venv
```

Active l'environnement virtuel :

### Windows
```bash
.venv\Scripts\activate
```

### macOS / Linux
```bash
source .venv/bin/activate
```

Installe les librairies :

```bash
pip install -r requirements.txt
```

## Lancer l'entraînement

```bash
python src/train.py
```

Le script va :

1. charger CIFAR-10 ;
2. normaliser les images ;
3. créer un CNN ;
4. entraîner le modèle ;
5. sauvegarder le modèle dans `models/cnn_cifar10.keras` ;
6. sauvegarder les courbes dans `results/training_curves.png`.

## Tester une prédiction

Après entraînement, tu peux tester une image :

```bash
python src/predict.py chemin/vers/image.jpg
```

## À expliquer dans ton rapport

Tu peux expliquer :

- pourquoi on utilise un CNN pour les images ;
- le rôle des couches `Conv2D` ;
- le rôle de `MaxPooling2D` ;
- la fonction de perte utilisée ;
- l'optimiseur Adam ;
- l'accuracy obtenue ;
- les limites du modèle.

## Pistes d'amélioration

- augmenter le nombre d'epochs ;
- ajouter de la data augmentation ;
- tester une architecture plus profonde ;
- ajouter du dropout pour limiter l'overfitting ;
- utiliser ton propre dataset.
