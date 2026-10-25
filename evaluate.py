
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "printqueue_data.csv"
PREDICTIONS_PATH = BASE_DIR / "reports" / "test_predictions.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
results = pd.read_csv(PREDICTIONS_PATH)

actual = results["actual_minutes"]
predicted = results["predicted_minutes"]
residuals = actual - predicted

# 1. Actual vs Predicted plot
plt.figure(figsize=(7, 5))
plt.scatter(actual, predicted, alpha=0.75)

minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())
plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")

plt.xlabel("Actual Completion Time (minutes)")
plt.ylabel("Predicted Completion Time (minutes)")
plt.title("Actual vs Predicted Completion Time")
plt.tight_layout()
plt.savefig(REPORT_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()

# 2. Residual plot
plt.figure(figsize=(7, 5))
plt.scatter(predicted, residuals, alpha=0.75)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Completion Time (minutes)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(REPORT_DIR / "residual_plot.png", dpi=300)
plt.close()

# 3. Numeric feature correlation heatmap
numeric_df = df[
    [
        "page_count",
        "jobs_ahead",
        "available_printers",
        "pages_ahead",
        "completion_time_minutes",
    ]
]

plt.figure(figsize=(8, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

# 4. Save evaluation metrics
metrics = pd.DataFrame(
    {
        "Metric": ["MAE (minutes)", "RMSE (minutes)", "R2 Score"],
        "Value": [
            mean_absolute_error(actual, predicted),
            mean_squared_error(actual, predicted) ** 0.5,
            r2_score(actual, predicted),
        ],
    }
)

metrics.to_csv(REPORT_DIR / "model_metrics.csv", index=False)

print("Evaluation completed successfully.")
print("\n", metrics.to_string(index=False))
print("\nGenerated files:")
print("- reports/actual_vs_predicted.png")
print("- reports/residual_plot.png")
print("- reports/correlation_heatmap.png")
print("- reports/model_metrics.csv")
