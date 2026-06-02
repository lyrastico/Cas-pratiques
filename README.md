# Cas pratiques — Deep Learning

Ce dépôt contient les cas pratiques réalisés dans le cadre du cours de Deep Learning.

L'objectif général est de mettre en application différentes architectures de réseaux de neurones sur des problèmes concrets :

* classification d'images avec un CNN ;
* classification de texte avec un Transformer.

## Structure du dépôt

```text
Cas_pratiques/
├── .venv/                         # Environnement virtuel local, non versionné
├── cas_pratique_1_cnn/             # TP 1 : classification d'images
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── results/
│   ├── src/
│   ├── requirements.txt
│   ├── notes_tp.md
│   └── README.md
├── cas_pratique_2_transformer/     # TP 2 : classification de texte
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── results/
│   ├── src/
│   ├── requirements.txt
│   ├── notes_tp.md
│   └── README.md
├── .gitignore
└── README.md
```

## Cas pratique n°1 — Classification d'images avec un CNN

Le premier cas pratique consiste à créer un modèle de Deep Learning capable de classifier des images à l'aide d'un réseau de neurones convolutif, aussi appelé CNN.

Le projet utilise le dataset CIFAR-10, composé de 60 000 images couleur de taille 32x32 pixels réparties en 10 classes :

* avion ;
* automobile ;
* oiseau ;
* chat ;
* cerf ;
* chien ;
* grenouille ;
* cheval ;
* bateau ;
* camion.

Le modèle utilisé contient plusieurs couches de convolution, des couches de pooling, une couche `Flatten` et des couches `Dense`.

Résultat obtenu sur les données de test :

```text
Accuracy test : environ 70,45 %
Loss test     : environ 0,8801
```

Le dossier du TP est disponible ici :

```text
cas_pratique_1_cnn/
```

## Cas pratique n°2 — Classification de texte avec un Transformer

Le deuxième cas pratique consiste à créer un modèle de Deep Learning capable de classifier automatiquement des textes.

Le modèle utilisé est basé sur une architecture Transformer. Il permet d'analyser les relations entre les mots d'une phrase grâce au mécanisme d'attention.

Le projet réalise une classification binaire de sentiments :

* `0` : avis négatif ;
* `1` : avis positif.

Le modèle contient une couche d'embedding, un bloc Transformer, une couche de pooling, des couches `Dropout` et des couches `Dense`.

Résultat obtenu sur les données de test :

```text
Accuracy test : environ 87,67 %
Loss test     : environ 0,2934
```

Le dossier du TP est disponible ici :

```text
cas_pratique_2_transformer/
```

## Installation générale

Depuis la racine du dépôt, créer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement virtuel :

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Installer les dépendances d'un TP :

```bash
pip install -r cas_pratique_1_cnn/requirements.txt
```

ou :

```bash
pip install -r cas_pratique_2_transformer/requirements.txt
```

## Lancer le cas pratique n°1

Depuis la racine du dépôt :

```bash
cd cas_pratique_1_cnn
python src/train.py
```

Le script entraîne le CNN, sauvegarde le modèle dans `models/` et génère les courbes d'entraînement dans `results/`.

Pour tester une prédiction :

```bash
python src/predict.py chemin/vers/image.jpg
```

## Lancer le cas pratique n°2

Depuis la racine du dépôt :

```bash
cd cas_pratique_2_transformer
python src/train.py
```

Le script entraîne le modèle Transformer, sauvegarde le modèle dans `models/` et génère les courbes d'entraînement dans `results/`.

Pour tester une prédiction :

```bash
python src/predict.py "ce film est vraiment excellent"
```

## Notions abordées

Ces cas pratiques permettent de travailler plusieurs notions importantes du Deep Learning :

* réseau de neurones convolutif ;
* convolution ;
* pooling ;
* classification multi-classes ;
* traitement du langage naturel ;
* embeddings ;
* mécanisme d'attention ;
* Transformer ;
* classification binaire ;
* overfitting ;
* EarlyStopping ;
* interprétation des courbes d'entraînement.

## Résultats globaux

| Cas pratique | Modèle      | Type de données | Type de classification | Accuracy test   |
| ------------ | ----------- | --------------- | ---------------------- | --------------- |
| TP 1         | CNN         | Images CIFAR-10 | Multi-classes          | environ 70,45 % |
| TP 2         | Transformer | Textes          | Binaire                | environ 87,67 % |

## Organisation Git

Le dossier `.venv/` ne doit pas être envoyé sur GitHub.
Les fichiers inutiles ou générés automatiquement doivent être ignorés avec le fichier `.gitignore`.

Exemple de `.gitignore` à la racine :

```gitignore
.venv/
*/.venv/
__pycache__/
*.pyc
data/
.DS_Store
```

## Conclusion

Ces deux cas pratiques montrent comment utiliser le Deep Learning sur deux types de données différents.

Le premier TP montre l'intérêt des CNN pour extraire automatiquement des caractéristiques visuelles à partir d'images.
Le deuxième TP montre l'intérêt des Transformers pour analyser du texte et prendre en compte les relations entre les mots.

Les résultats obtenus sont satisfaisants pour des modèles de base et peuvent être améliorés avec des architectures plus avancées, davantage de données ou un réglage plus fin des hyperparamètres.
