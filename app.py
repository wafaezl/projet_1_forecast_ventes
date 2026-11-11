from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
st.set_page_config(page_title="Retail Intelligence", layout="wide")
st.title("Retail Intelligence — Analyse & prévision des ventes")
st.caption("Projet portfolio Data Analyst & Machine Learning | Données synthétiques")

required = [DATA / "transactions.csv", DATA / "daily_sales.csv", DATA / "forecast.csv"]
if not all(p.exists() for p in required):
    st.warning("Données absentes. Lancez d'abord les commandes du README.")
    st.stop()

df = pd.read_csv(DATA / "transactions.csv", parse_dates=["date"])
daily = pd.read_csv(DATA / "daily_sales.csv", parse_dates=["date"])
forecast = pd.read_csv(DATA / "forecast.csv", parse_dates=["date"])
metrics_path = DATA / "metrics.json"
metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

total = df["revenue"].sum()
orders = df["transaction_id"].nunique()
aov = df.groupby("transaction_id")["revenue"].sum().mean()
c1, c2, c3 = st.columns(3)
c1.metric("Chiffre d'affaires", f"{total:,.0f} €")
c2.metric("Transactions", f"{orders:,.0f}")
c3.metric("Panier moyen", f"{aov:,.2f} €")

st.subheader("Évolution quotidienne du chiffre d'affaires")
st.line_chart(daily.set_index("date")["revenue"])

st.subheader("Prévision vs réalité — période de test")
chart = forecast.set_index("date")[["actual_revenue", "predicted_revenue"]]
st.line_chart(chart)
if metrics:
    st.write("Métriques du modèle :", metrics)

st.subheader("Performance par catégorie")
category = df.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
st.bar_chart(category.set_index("category"))

st.subheader("Filtres d'analyse")
selected = st.multiselect("Canal de vente", sorted(df["channel"].unique()),
                          default=sorted(df["channel"].unique()))
filtered = df[df["channel"].isin(selected)]
st.dataframe(filtered.groupby("category").agg(
    chiffre_affaires=("revenue", "sum"),
    quantites=("quantity", "sum"),
    transactions=("transaction_id", "nunique")
).sort_values("chiffre_affaires", ascending=False), use_container_width=True)
