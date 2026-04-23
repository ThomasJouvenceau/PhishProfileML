# Data Pipeline

## Objectif

Cette partie couvre la preparation des profils avant la segmentation ML.
Le flux de travail est le suivant :

1. `data/raw_profiles.csv` contient les profils bruts locaux.
2. `src/preprocess.py` nettoie, valide et encode les variables.
3. `data/processed_profiles.csv` contient le dataset numerique partageable.
4. `src/main.py` consomme ensuite ce dataset pour la segmentation et la recommandation.

## Fichiers de donnees

- `data/raw_profiles.csv` : fichier source local, ignore par Git pour eviter de pousser des donnees potentiellement sensibles.
- `data/profiles_template.csv` : jeu d'exemple deja present dans le repo, utilise comme repli si `raw_profiles.csv` n'existe pas.
- `data/processed_profiles.csv` : dataset encode, pret pour le pipeline ML.
- `data/campaign_results.csv` : trame locale pour enregistrer les resultats d'ouverture, clic, reponse et delai.

## Valeurs acceptees

Les colonnes qualitatives suivantes sont gerees :

- `statut` : `initial`, `alternance`
- `specialite_dominante` : `cyber`, `dev`, `data`, `mixte`
- `style_communication` : `formel`, `mixte`, `informel`
- colonnes ordinales : `faible`, `moyen`, `moyenne`, `fort`, `forte`

Le script accepte aussi les valeurs deja encodees (`0`, `1`, `2`, `3`) pour reutiliser les fichiers existants.

## Commandes utiles

Preparer uniquement le dataset :

```bash
python src/preprocess.py
```

Lancer la pipeline complete :

```bash
python src/main.py
```

## Controles integres

Le preprocessing verifie automatiquement :

- la presence des colonnes obligatoires ;
- les identifiants vides ou dupliques ;
- les valeurs non reconnues dans les colonnes categorielle et ordinales.
