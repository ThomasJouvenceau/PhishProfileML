# PhishProfileML

## Description

PhishProfileML est un projet de machine learning appliqué à la cybersécurité.  
L’objectif est de développer un outil capable de segmenter des profils étudiants et de recommander des catégories de messages personnalisés afin d’optimiser et d’évaluer l’efficacité d’une campagne email dans un cadre académique.

Le projet combine :
- collecte et structuration de profils ;
- création d’un dataset exploitable ;
- segmentation par machine learning ;
- recommandation de catégories de messages ;
- comparaison entre une approche générique et une approche personnalisée ;
- analyse des résultats obtenus.

## Contexte

Ce projet est réalisé dans le cadre du cours **Machine Learning for Cybersecurity**.

Les livrables attendus sont :
- une vidéo YouTube privée de 5 à 15 minutes détaillant le projet ;
- un repository GitHub privé contenant le code complet du projet.

## Objectif du projet

L’objectif principal est de démontrer qu’un outil de machine learning peut aider à trier intelligemment des profils afin d’adapter une campagne email à chaque segment identifié.

Le projet cherche à montrer que la personnalisation pilotée par le ML peut améliorer les performances observées par rapport à une campagne générique.

## Hypothèse

L’hypothèse principale est la suivante :

> Une campagne personnalisée à partir d’une segmentation ML obtient de meilleurs résultats qu’une campagne générique, notamment sur les taux d’ouverture, de clic et de réponse.

## Population étudiée

Le panel étudié est composé de **26 étudiants de dernière année de bachelor**, appartenant à la même classe.

Les résultats seront analysés dans le cadre du projet, puis restitués de manière anonymisée à travers des identifiants de profil et des profils types.

## Méthodologie générale

Le projet suit les étapes suivantes :

1. Cadrage du projet et définition de l’hypothèse.
2. Construction de fiches profils anonymisées.
3. Création d’un dataset structuré.
4. Encodage et préparation des données.
5. Segmentation des profils avec du machine learning.
6. Recommandation d’une catégorie de message adaptée.
7. Comparaison entre campagne générique et campagne personnalisée.
8. Analyse des résultats.
9. Restitution via GitHub et vidéo de démonstration.

## Variables utilisées

Chaque profil est représenté par plusieurs variables standardisées :

- `id_profil`
- `statut`
- `specialite_dominante`
- `implication_academique`
- `interet_contenu_pedagogique`
- `implication_vie_etudiante`
- `interet_evenementiel`
- `niveau_technique_estime`
- `aisance_numerique`
- `style_communication`
- `reactivite_percue`
- `presence_numerique_visible`

Les valeurs numériques sont généralement codées ainsi :

- `1` = faible
- `2` = moyen
- `3` = fort

## Segmentation des profils

Le projet prévoit une segmentation en plusieurs familles de profils, par exemple :

### 1. Profil académique

Profil plutôt orienté cours, rendus, contenus pédagogiques et informations institutionnelles.

### 2. Profil associatif / événementiel

Profil sensible à la vie de promotion, aux événements, aux interactions de groupe et aux communications collectives.

### 3. Profil technique / vigilant

Profil plus à l’aise avec les outils numériques et les sujets techniques.

### 4. Profil hybride / social

Profil polyvalent, socialement visible, avec plusieurs centres d’intérêt.

## Pipeline technique

Le pipeline actuel comprend :

1. Chargement du fichier CSV de profils.
2. Encodage des variables catégorielles.
3. Normalisation des données.
4. Segmentation des profils avec KMeans.
5. Recommandation d’une catégorie de message.
6. Export des résultats dans un fichier CSV.
7. Analyse future des résultats expérimentaux.

## Structure du repository

```text
phishprofile-ml/
├── data/
│   ├── profiles_template.csv
│   ├── raw_profiles.csv
│   ├── processed_profiles.csv
│   └── campaign_results.csv
├── docs/
├── models/
├── notebooks/
├── results/
│   ├── charts/
│   ├── tables/
│   └── profiles_segmented.csv
├── src/
│   ├── preprocess.py
│   ├── segment_profiles.py
│   ├── recommend_category.py
│   ├── evaluate_results.py
│   └── main.py
├── .gitignore
├── README.md
└── requirements.txt