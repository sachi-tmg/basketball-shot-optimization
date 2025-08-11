import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# === Directories and files ===
input_file = "cleaned_data/nba_playoff_2015_2024_structured.csv"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

lr_model_file = os.path.join(model_dir, "logistic_regression_model.joblib")
rf_model_file = os.path.join(model_dir, "random_forest_model.joblib")
encoder_file = os.path.join(model_dir, "shooter_label_encoder.joblib")
model_ready_file = "model_ready_shots.csv"

# === Load data ===
df = pd.read_csv(input_file)

# === Feature engineering ===
df['shot_made_flag'] = df['shot_result'].map({'make': 1, 'miss': 0})

# 1 = assisted, 0 = not assisted
df['was_assisted'] = (df['assisted_by'].notna() & (df['assisted_by'] != 'Unassisted')).astype(int)
df['was_blocked'] = (df['blocked_by'].notna() & (df['blocked_by'] != 'Not Blocked')).astype(int)

# One-hot encode shot_type
df = pd.get_dummies(df, columns=["shot_type"], prefix="type")

# Numeric period
df['period_num'] = pd.to_numeric(df['period'], errors='coerce')

# Seconds left (from MM:SS.S format)
def time_to_seconds(t):
    try:
        parts = str(t).split(":")
        if len(parts) == 2:
            minutes, seconds = float(parts[0]), float(parts[1])
            return int(minutes * 60 + seconds)
        else:
            return np.nan
    except:
        return np.nan

df['seconds_left'] = df['time_left'].apply(time_to_seconds)

# Score margin (parse if exists, else fillna)
def parse_score(score):
    try:
        away, home = map(int, str(score).split("-"))
        return pd.Series([away, home, home - away])
    except:
        return pd.Series([np.nan, np.nan, np.nan])

df[['away_score', 'home_score', 'score_margin']] = df['score'].apply(parse_score)

# Encode shooter name
shooter_encoder = LabelEncoder()
df['shooter_encoded'] = shooter_encoder.fit_transform(df['shooter'].fillna("Unknown"))

# === Remove rows with missing values in key features ===
required = ['shot_distance', 'shot_made_flag', 'period_num', 'seconds_left', 'score_margin']
print("Rows before filtering:", len(df))
print("Missing values in required columns:\n", df[required].isna().sum())
df_model = df.dropna(subset=required)
print("Rows after filtering:", len(df_model))

# === Select modeling columns ===
feature_cols = ['shooter_encoded', 'shot_distance', 'period_num', 'seconds_left',
                'score_margin', 'was_assisted', 'was_blocked', 'shot_made_flag'] + \
               [c for c in df_model.columns if c.startswith("type_")]

df_model[feature_cols].to_csv(model_ready_file, index=False)
joblib.dump(shooter_encoder, encoder_file)

# === Modeling ===
X = df_model[feature_cols].drop("shot_made_flag", axis=1)
y = df_model["shot_made_flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Save models
joblib.dump(lr, lr_model_file)
joblib.dump(rf, rf_model_file)

# Evaluation
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# Logistic regression feature importance
print("\nFeature importances (Logistic Regression):")
for name, coef in zip(X.columns, lr.coef_[0]):
    print(f"{name}: {coef:.4f}")

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Random forest feature importances
print("\nFeature importances (Random Forest):")
importances = rf.feature_importances_
for name, imp in sorted(zip(X.columns, importances), key=lambda x: -x[1]):
    print(f"{name}: {imp:.4f}")

