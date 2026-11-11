# Analyse prédictive des ventes et prévision de la demande

Projet portfolio Data Analyst & IA : génération d'un jeu de données commercial synthétique,
analyse exploratoire, prévision des ventes et tableau de bord interactif.

## Fonctionnalités
- Génération de 50 000 transactions synthétiques sur deux ans.
- Analyse des ventes, tendances, saisonnalités et catégories.
- Prévision du chiffre d'affaires quotidien avec un modèle de régression
  utilisant des variables calendaires, retards et moyennes mobiles.
- Évaluation sur les 30 derniers jours avec MAE, RMSE et MAPE.
- Dashboard Streamlit présentant les KPI, tendances et prévisions.

## Installation
```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
# macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution
```bash
python src/generate_data.py
python src/analysis.py
python src/train_forecast.py
streamlit run app.py
```

Les fichiers CSV sont générés dans `data/`. Les résultats du modèle sont enregistrés
dans `data/forecast.csv` et `data/metrics.json`.

## À personnaliser
Remplacez le générateur synthétique par vos propres données anonymisées. Les résultats
obtenus sur les données synthétiques ne doivent pas être présentés comme des performances
obtenues en entreprise.
