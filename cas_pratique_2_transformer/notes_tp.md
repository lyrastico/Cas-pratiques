# Cas pratique n°2 — Classification de texte avec un Transformer

## Objectif

L'objectif de ce cas pratique est de créer un modèle de Deep Learning capable de classifier automatiquement des textes à l'aide d'une architecture de type Transformer.

Le modèle utilisé effectue une classification de sentiment sur des avis de films. Il doit prédire si un avis est positif ou négatif.

## Dataset utilisé

Le projet utilise le dataset IMDB fourni par TensorFlow/Keras. Ce dataset contient des avis de films associés à une étiquette binaire :

- 0 : avis négatif ;
- 1 : avis positif.

Les textes sont convertis en séquences numériques afin de pouvoir être utilisés par le modèle.

## Architecture du modèle

Le modèle contient :

- une couche d'embedding pour transformer les mots en vecteurs ;
- une couche d'encodage positionnel ;
- un bloc Transformer avec Multi-Head Attention ;
- une couche de normalisation ;
- un réseau dense final pour la classification binaire.

## Entraînement

Le modèle est entraîné sur plusieurs epochs avec une validation automatique pendant l'entraînement.

Paramètres utilisés :

- vocabulaire maximal : 20 000 mots ;
- longueur maximale des séquences : 200 tokens ;
- batch size : 64 ;
- optimizer : Adam ;
- fonction de perte : binary_crossentropy.

## Résultats

À compléter après l'entraînement :

- accuracy entraînement :
- loss entraînement :
- accuracy validation :
- loss validation :
- accuracy test :
- loss test :

## Analyse

À compléter après l'entraînement.

On pourra comparer l'évolution de l'accuracy et de la loss grâce au graphique généré dans `results/training_curves.png`.

## Conclusion

Ce cas pratique montre comment utiliser une architecture de type Transformer pour traiter du texte. Contrairement aux CNN utilisés pour les images, les Transformers sont particulièrement adaptés aux séquences, notamment grâce au mécanisme d'attention qui permet au modèle de se concentrer sur les mots importants d'une phrase.
