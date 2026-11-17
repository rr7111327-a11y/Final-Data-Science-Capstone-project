"""
Task 6 - Final Data Science Capstone Project
Student Performance Analysis and Final Grade Prediction

Run:
    python -m pip install -r requirements.txt
    python main.py

The included CSV is a clearly labelled synthetic DEMO dataset for reproducible practice.
For a research-grade submission, replace it with the original UCI Student Performance
Mathematics CSV (student-mat.csv) from https://archive.ics.uci.edu/dataset/320/student+performance
and save it as data/student-mat.csv. The script supports both formats.
"""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def load_data():
    original = DATA / "student-mat.csv"
    demo = DATA / "student_performance_demo.csv"
    path = original if original.exists() else demo
    if not path.exists():
        raise FileNotFoundError("Place student-mat.csv or student_performance_demo.csv in data/.")
    df = pd.read_csv(path, sep=";" if path.name == "student-mat.csv" else ",")
    df.columns = [c.strip() for c in df.columns]
    if "G3" not in df.columns:
        raise ValueError("Dataset must contain target column G3.")
    return df, path.name

def main():
    df, source_name = load_data()
    print(f"Loaded {len(df)} records from {source_name}")
    print("\\n--- Data quality ---")
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=["G3"])
    df["G3"] = pd.to_numeric(df["G3"], errors="coerce")
    df = df.dropna(subset=["G3"])
    df = df[(df["G3"] >= 0) & (df["G3"] <= 20)]
    if len(df) < 30:
        raise ValueError("At least 30 usable records are required for this demo model.")

    # Exclude G1/G2 for an early-support model? Here they are retained as prior-period
    # grades where available; this is a late-term prediction and not a causal model.
    X = df.drop(columns=["G3"])
    y = df["G3"]
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    pipeline = Pipeline([("preprocessor", preprocessor),
                         ("regressor", RandomForestRegressor(
                             n_estimators=300, random_state=42, min_samples_leaf=2))])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "records": int(len(df)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "MAE": round(float(mean_absolute_error(y_test, predictions)), 3),
        "RMSE": round(float(mean_squared_error(y_test, predictions) ** 0.5), 3),
        "R2": round(float(r2_score(y_test, predictions)), 3)
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"actual_G3": y_test.to_numpy(),
                  "predicted_G3": predictions.round(2)}).to_csv(
                      OUT / "predictions.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.hist(df["G3"], bins=11, edgecolor="black")
    plt.title("Distribution of Final Grades (G3)")
    plt.xlabel("Final grade (0–20)"); plt.ylabel("Number of records")
    plt.tight_layout(); plt.savefig(OUT / "grade_distribution.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.plot([0, 20], [0, 20], linestyle="--")
    plt.title("Actual vs Predicted Final Grades")
    plt.xlabel("Actual G3"); plt.ylabel("Predicted G3")
    plt.tight_layout(); plt.savefig(OUT / "actual_vs_predicted.png", dpi=150); plt.close()

    print("\\n--- Model evaluation (test set) ---")
    for key, value in metrics.items(): print(f"{key}: {value}")
    print(f"\\nOutputs saved in: {OUT}")
    print("Note: predictions are estimates for learning support, not decisions about students.")

if __name__ == "__main__":
    main()
