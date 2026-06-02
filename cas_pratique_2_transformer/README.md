# Cas pratique n°2 — Classification de texte avec un Transformer

Projet de base pour commencer le TP sur VS Code.

## Objectif

Créer un modèle de Deep Learning capable de classifier automatiquement des textes à l'aide d'une architecture de type Transformer.

Par défaut, le projet utilise le dataset **IMDB** intégré à TensorFlow/Keras. Il contient des avis de films classés en deux catégories :

- avis négatif ;
- avis positif.

Le modèle apprend donc à faire une classification binaire de sentiment.

## Structure du dossier

```text
cas_pratique_2_transformer/
├── data/
├── models/
├── notebooks/
├── results/
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── .gitignore
├── notes_tp.md
├── README.md
└── requirements.txt
```

## Installation

Depuis le dossier qui contient les deux cas pratiques, tu peux utiliser un environnement virtuel commun :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r cas_pratique_2_transformer\requirements.txt
```

Ou directement depuis ce dossier :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Lancer l'entraînement

Depuis le dossier `cas_pratique_2_transformer` :

```powershell
python src\train.py
```

Le script entraîne le modèle, l'évalue sur les données de test, puis sauvegarde :

- le modèle dans `models/transformer_imdb.keras` ;
- les courbes dans `results/training_curves.png`.

## Tester une phrase

Après l'entraînement :

```powershell
python src\predict.py "This movie was really good and emotional"
```

Exemple en français traduit approximativement en anglais, car le dataset IMDB est en anglais :

```powershell
python src\predict.py "The movie was boring and too long"
```

## Idées d'amélioration

- Augmenter le nombre d'epochs.
- Ajouter de l'EarlyStopping.
- Modifier la taille de l'embedding.
- Modifier le nombre de têtes d'attention.
- Tester un autre dataset de texte.
- Ajouter une matrice de confusion.
