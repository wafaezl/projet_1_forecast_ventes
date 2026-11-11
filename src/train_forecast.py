from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def make_features(series):
    frame = series.rename("revenue").to_frame()
    idx = frame.index
    frame["dayofweek"] = idx.dayofweek
    frame["month"] = idx.month
    frame["dayofyear"] = idx.dayofyear
    frame["weekofyear"] = idx.isocalendar().week.astype(int)
    frame["lag_1"] = frame["revenue"].shift(1)
    frame["lag_7"] = frame["revenue"].shift(7)
    frame["lag_14"] = frame["revenue"].shift(14)
    frame["rolling_mean_7"] = frame["revenue"].shift(1).rolling(7).mean()
    frame["rolling_mean_28"] = frame["revenue"].shift(1).rolling(28).mean()
    return frame

def main():
    daily = pd.read_csv(DATA / "daily_sales.csv", parse_dates=["date"])
    series = daily.set_index("date")["revenue"].asfreq("D").fillna(0)
    features = make_features(series).dropna()
    X = features.drop(columns="revenue")
    y = features["revenue"]

    cutoff = y.index.max() - pd.Timedelta(days=29)
    X_train, X_test = X.loc[X.index < cutoff], X.loc[X.index >= cutoff]
    y_train, y_test = y.loc[y.index < cutoff], y.loc[y.index >= cutoff]

    model = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, max_leaf_nodes=20)
    model.fit(X_train, y_train)
    pred = np.maximum(model.predict(X_test), 0)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    nonzero = y_test != 0
    mape = (np.abs((y_test[nonzero] - pred[nonzero]) / y_test[nonzero]).mean() * 100
            if nonzero.any() else None)

    result = pd.DataFrame({"date": y_test.index, "actual_revenue": y_test.values,
                           "predicted_revenue": pred})
    result.to_csv(DATA / "forecast.csv", index=False)
    metrics = {"MAE": float(mae), "RMSE": float(rmse),
               "MAPE_percent": float(mape) if mape is not None else None}
    (DATA / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Évaluation sur les 30 derniers jours :", metrics)

if __name__ == "__main__":
    main()
