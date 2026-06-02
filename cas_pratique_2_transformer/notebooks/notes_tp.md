# Notes pour le rapport

## Introduction

Ce projet consiste à réaliser une classification de texte à l'aide d'un modèle Transformer.

L'objectif est de créer un modèle de Deep Learning capable d'analyser automatiquement un texte et de déterminer sa classe. Dans ce cas pratique, le modèle effectue une classification binaire : il prédit si un avis est positif ou négatif.

## Pourquoi un Transformer ?

Un Transformer est adapté au traitement du langage naturel car il permet de prendre en compte les relations entre les mots d'une phrase.

Contrairement à un réseau de neurones classique, le Transformer utilise un mécanisme appelé attention. Ce mécanisme permet au modèle d'identifier les mots importants dans une phrase et de comprendre leur influence sur le sens global du texte.

Dans ce projet, le Transformer permet donc d'apprendre automatiquement les caractéristiques importantes des textes afin de réaliser une classification.

## Dataset utilisé

Le modèle a été entraîné sur un dataset de textes destiné à la classification binaire de sentiments.

Ce type de dataset contient généralement des avis textuels associés à une étiquette :

* `0` pour un avis négatif ;
* `1` pour un avis positif.

Dans ce projet, les textes sont transformés en séquences numériques afin de pouvoir être utilisés par le modèle. Chaque mot est représenté par un identifiant numérique, puis les séquences sont limitées à une longueur maximale de 200 tokens.

Une partie des données d'entraînement est utilisée comme données de validation afin de suivre les performances du modèle pendant l'apprentissage.

## Architecture utilisée

Le modèle contient :

* une couche d'entrée recevant des séquences de 200 tokens ;
* une couche `TokenAndPositionEmbedding` pour représenter les mots et leur position dans la phrase ;
* un bloc `TransformerBlock` basé sur le mécanisme d'attention ;
* une couche `GlobalAveragePooling1D` pour résumer les informations de la séquence ;
* des couches `Dropout` pour limiter le surapprentissage ;
* une couche `Dense` cachée de 64 neurones ;
* une couche finale `Dense` avec 1 neurone pour effectuer une classification binaire.

La couche finale contient un seul neurone, car le modèle doit prédire une probabilité entre 0 et 1.
Une valeur proche de 0 correspond à un avis négatif, tandis qu'une valeur proche de 1 correspond à un avis positif.

## Entraînement

Le modèle est entraîné avec l'optimiseur Adam et une fonction de perte adaptée à la classification binaire.

L'entraînement était prévu sur 10 epochs, mais un mécanisme d'EarlyStopping a été utilisé afin d'éviter le surapprentissage.

L'EarlyStopping surveille la perte de validation. Si cette perte ne s'améliore plus pendant plusieurs epochs, l'entraînement s'arrête automatiquement et les meilleurs poids du modèle sont restaurés.

## Résultats obtenus

Le modèle Transformer a été entraîné sur un problème de classification binaire de texte.

Résultats finaux :

* Accuracy entraînement : 98,11 %
* Loss entraînement : 0,0590
* Accuracy validation : 86,86 %
* Loss validation : 0,4766
* Accuracy test : 87,67 %
* Loss test : 0,2934

Ces résultats montrent que le modèle apprend très rapidement à classifier les textes.
L'accuracy d'entraînement passe d'environ 78,68 % à 98,11 % en seulement 4 epochs.

L'accuracy de validation atteint son meilleur niveau dès la première epoch avec environ 88,24 %. Ensuite, elle diminue légèrement, ce qui montre que le modèle commence à surapprendre les données d'entraînement.

## Architecture du modèle

Le modèle utilisé est un modèle Transformer composé de :

* 1 couche d'entrée de taille 200 ;
* 1 couche `TokenAndPositionEmbedding` ;
* 1 bloc `TransformerBlock` ;
* 1 couche `GlobalAveragePooling1D` ;
* 2 couches `Dropout` ;
* 1 couche `Dense` cachée de 64 neurones ;
* 1 couche finale `Dense` avec 1 neurone.

Le modèle contient au total 1 338 817 paramètres entraînables.

La couche `TokenAndPositionEmbedding` permet de représenter les mots ainsi que leur position dans la phrase.
Le bloc Transformer permet d'analyser les relations entre les mots grâce au mécanisme d'attention.
La couche `GlobalAveragePooling1D` transforme la séquence en un vecteur unique utilisable pour la classification.
Les couches `Dropout` permettent de réduire le risque de surapprentissage.
La couche finale permet d'obtenir une probabilité pour la classification binaire.

![alt text](image.png)

![alt text](image-1.png)

## Entraînement avec EarlyStopping

Une expérience a été réalisée avec un maximum de 10 epochs et un mécanisme d'EarlyStopping.
L'objectif était de laisser le modèle apprendre tout en évitant qu'il ne surapprenne les données d'entraînement.

L'entraînement s'est arrêté automatiquement à l'epoch 4, car la loss de validation ne s'améliorait plus suffisamment.

Le meilleur niveau de validation a été observé dès la première epoch, avec :

* Accuracy validation : 88,24 %
* Loss validation : 0,2867

Le modèle final obtient une accuracy de test de 87,67 %.
Ce résultat montre que le modèle généralise correctement sur des textes qu'il n'a pas vus pendant l'entraînement.

On observe cependant un début de surapprentissage : l'accuracy d'entraînement continue d'augmenter jusqu'à 98,11 %, tandis que l'accuracy de validation diminue légèrement après la première epoch.
L'EarlyStopping permet donc d'arrêter l'entraînement avant que le modèle ne se spécialise trop sur les données d'entraînement.

![alt text](image-2.png)

## Analyse

Les résultats obtenus sont satisfaisants pour un premier modèle Transformer.

Le modèle atteint environ 87,67 % de bonnes prédictions sur les données de test. Cela montre que l'architecture Transformer est efficace pour traiter du texte et effectuer une classification de sentiments.

Cependant, l'écart entre l'accuracy d'entraînement et l'accuracy de validation indique un début de surapprentissage. Le modèle apprend très rapidement les données d'entraînement, mais ses performances sur la validation ne progressent plus après la première epoch.

Pour améliorer le modèle, il serait possible de :

* augmenter le taux de Dropout ;
* réduire la taille du modèle ;
* utiliser plus de données d'entraînement ;
* tester différents paramètres comme la taille des embeddings ou le nombre de têtes d'attention ;
* ajuster le learning rate ;
* utiliser un modèle pré-entraîné comme BERT.

## Conclusion provisoire

Le Transformer permet d'apprendre automatiquement les relations entre les mots d'un texte grâce au mécanisme d'attention.

Dans ce cas pratique, le modèle obtient une accuracy de test de 87,67 %, ce qui montre qu'il est capable de classifier efficacement des avis positifs et négatifs.

L'utilisation de l'EarlyStopping est pertinente, car elle permet de limiter le surapprentissage et de conserver les meilleurs poids du modèle.
