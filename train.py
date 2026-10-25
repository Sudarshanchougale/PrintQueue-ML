
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "printqueue_data.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load and validate the dataset
df = pd.read_csv(DATA_PATH)

target = "completion_time_minutes"

numeric_features = [
    "page_count",
    "jobs_ahead",
    "available_printers",
    "pages_ahead",
]

categorical_features = [
    "printing_type",
    "binding_required",
]

required_columns = numeric_features + categorical_features + [target]

missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

df = df.dropna(subset=required_columns).copy()

if len(df) < 10:
    raise ValueError("Not enough valid rows to train the model.")

X = df[numeric_features + categorical_features]
y = df[target]


# 3. Split data before fitting preprocessing or the model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# 4. Preprocess numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)


# 5. Create and train the regression pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

model.fit(X_train, y_train)


# 6. Evaluate on unseen test data
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n===== PrintQueue Model Evaluation =====")
print(f"Total dataset rows: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"MAE:  {mae:.2f} minutes")
print(f"RMSE: {rmse:.2f} minutes")
print(f"R²:   {r2:.3f}")


# 7. Save the complete pipeline for future predictions
model_path = MODEL_DIR / "printqueue_model.joblib"
joblib.dump(model, model_path)

# Save test predictions for analysis and plotting
results = pd.DataFrame(
    {
        "actual_minutes": y_test,
        "predicted_minutes": predictions,
    },
    index=y_test.index,
).sort_index()

results.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

print(f"\nSaved model to: {model_path}")
print(f"Saved test predictions to: {REPORT_DIR / 'test_predictions.csv'}")
