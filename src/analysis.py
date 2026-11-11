from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "transactions.csv", parse_dates=["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)

daily = df.groupby("date", as_index=False).agg(
    revenue=("revenue", "sum"),
    orders=("transaction_id", "nunique"),
    units=("quantity", "sum")
)
monthly = df.groupby("month", as_index=False).agg(
    revenue=("revenue", "sum"),
    orders=("transaction_id", "nunique")
)
category = df.groupby("category", as_index=False).agg(
    revenue=("revenue", "sum"),
    units=("quantity", "sum")
).sort_values("revenue", ascending=False)

daily.to_csv(ROOT / "data" / "daily_sales.csv", index=False)
monthly.to_csv(ROOT / "data" / "monthly_sales.csv", index=False)
category.to_csv(ROOT / "data" / "category_performance.csv", index=False)

print("Chiffre d'affaires total :", round(df["revenue"].sum(), 2))
print("Panier moyen :", round(df.groupby("transaction_id")["revenue"].sum().mean(), 2))
print("\nPerformance par catégorie :")
print(category.to_string(index=False))
