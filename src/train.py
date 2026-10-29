"""Train and compare regression models on the processed insurance data."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

from data_processing import PROCESSED_PATH, ROOT, clean_and_process_data

MODEL_PATH = ROOT / "models" / "insurance_model.joblib"


def main():
    if not Path(PROCESSED_PATH).exists():
        clean_and_process_data()
    df = pd.read_csv(PROCESSED_PATH)
    X, y = df.drop(columns="charges"), df["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear": LinearRegression(),
        "Linear (log target)": TransformedTargetRegressor(
            LinearRegression(), func=np.log1p, inverse_func=np.expm1),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42),
        "GradBoost": GradientBoostingRegressor(max_depth=3, random_state=42),
    }

    best_name, best_model, best_r2 = None, None, -np.inf
    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        print(f"{name:20s} CV R2={cv:.3f} | test R2={r2:.3f} "
              f"MAE={mean_absolute_error(y_test, pred):,.0f} "
              f"RMSE={root_mean_squared_error(y_test, pred):,.0f}")
        if r2 > best_r2:
            best_name, best_model, best_r2 = name, model, r2

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "columns": list(X.columns)}, MODEL_PATH)
    print(f"Saved best model ({best_name}) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
