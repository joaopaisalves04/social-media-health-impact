"""
Run this script ONCE locally to train the model and save model_bundle.pkl.
The CSV file must be in the same directory.

Usage:
    python train_and_save_model.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

# Load dataset (must be in the same folder)
df = pd.read_csv("Social_media_impact_on_life.csv")
df_model = df.drop(columns=["Student_ID"]).copy()

# --- Target encoding ---
target_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
df_model["target"] = df_model["Overall_Impact"].map(target_map)

# --- Binary / ordinal encodings ---
df_model["Gender_enc"]    = (df_model["Gender"] == "Female").astype(int)
df_model["AcadPerf_enc"]  = (df_model["Affects_Academic_Performance"] == "Yes").astype(int)
df_model["AcadLevel_enc"] = df_model["Academic_Level"].map(
    {"High School": 0, "Undergraduate": 1, "Graduate": 2}
)

# --- One-hot: top 7 platforms, top 5 countries ---
top_platforms = df_model["Most_Used_Platform"].value_counts().nlargest(7).index
top_countries = df_model["Country"].value_counts().nlargest(5).index

df_model["Platform_grp"] = df_model["Most_Used_Platform"].where(
    df_model["Most_Used_Platform"].isin(top_platforms), "Other"
)
df_model["Country_grp"] = df_model["Country"].where(
    df_model["Country"].isin(top_countries), "Other"
)

platform_dummies = pd.get_dummies(df_model["Platform_grp"], prefix="plt")
country_dummies  = pd.get_dummies(df_model["Country_grp"],  prefix="ctry")

# --- Engineered features ---
median_usage = df_model["Avg_Daily_Usage_Hours"].median()

df_model["UsageSleep_ratio"]    = df_model["Avg_Daily_Usage_Hours"] / (df_model["Sleep_Hours_Per_Night"] + 1e-5)
df_model["HealthSleep_product"] = df_model["Mental_Health_Score"] * df_model["Sleep_Hours_Per_Night"]
df_model["HighUsage"]           = (df_model["Avg_Daily_Usage_Hours"] > median_usage).astype(int)

# --- Assemble feature matrix ---
base_features = [
    "Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Mental_Health_Score",
    "Gender_enc", "AcadPerf_enc", "AcadLevel_enc",
    "UsageSleep_ratio", "HealthSleep_product", "HighUsage"
]
X = pd.concat([df_model[base_features], platform_dummies, country_dummies], axis=1)
y = df_model["target"]

print(f"Feature columns ({len(X.columns)}): {list(X.columns)}")
print(f"Median usage threshold: {median_usage}")
print(f"Top platforms: {list(top_platforms)}")
print(f"Top countries: {list(top_countries)}")

# --- Train / test split (matches notebook) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train best model (best params from notebook GridSearchCV) ---
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
model.fit(X_train.values, y_train)

# Verify performance
y_pred = model.predict(X_test.values)
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"\nTest F1 (weighted): {f1:.4f}")

# --- Save bundle ---
bundle = {
    "model": model,
    "feature_columns": list(X.columns),
    "median_usage": float(median_usage),
    "top_platforms": list(top_platforms),
    "top_countries": list(top_countries),
    "label_map_inv": {0: "Negative", 1: "Neutral", 2: "Positive"}
}
joblib.dump(bundle, "model_bundle.pkl")
print("\nSaved model_bundle.pkl")
