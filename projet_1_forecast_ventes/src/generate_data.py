from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

def main():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    products = ["Maison", "Informatique", "Beauté", "Sport", "Mode"]
    n = 50_000

    transaction_dates = rng.choice(dates, size=n)
    categories = rng.choice(products, size=n, p=[.20, .22, .18, .17, .23])
    prices = {
        "Maison": 45, "Informatique": 180, "Beauté": 28,
        "Sport": 65, "Mode": 52
    }
    quantity = rng.choice([1, 2, 3, 4], size=n, p=[.62, .25, .10, .03])
    unit_price = np.array([prices[c] for c in categories]) * rng.uniform(.75, 1.3, n)
    seasonality = np.array([
        1 + .18 * np.sin(2 * np.pi * d.timetuple().tm_yday / 365.25)
        for d in pd.to_datetime(transaction_dates)
    ])
    revenue = unit_price * quantity * seasonality

    df = pd.DataFrame({
        "transaction_id": [f"T{i:06d}" for i in range(n)],
        "date": pd.to_datetime(transaction_dates),
        "category": categories,
        "quantity": quantity,
        "unit_price": unit_price.round(2),
        "revenue": revenue.round(2),
        "channel": rng.choice(["Web", "Magasin", "Mobile"], n, p=[.48, .32, .20])
    })
    df.to_csv(DATA / "transactions.csv", index=False)
    print(f"{len(df):,} transactions écrites dans data/transactions.csv")

if __name__ == "__main__":
    main()
