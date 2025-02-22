import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Load dataset
data_path = r"C:\Users\DCAPARRO-LT\Downloads\Anomaly Detection\electricity_consumption_mock_data.csv"  # Use raw string to prevent issues with backslashes
df = pd.read_csv(data_path)

# Prepare features and labels
X = df[["Month", "Monthly_Consumption_kWh"]].values  # Features (Month, Consumption)
y_true = df["Is_Anomalous"].values  # True labels for evaluation

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_true, test_size=0.3, random_state=42
)

# Train Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_train)

# Save the trained model to a file
model_path = "isolation_forest_model.pkl"
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

# Predict on the test set
y_pred = model.predict(X_test)
# Convert predictions: 1 (normal) -> 0, -1 (anomaly) -> 1
y_pred = np.where(y_pred == -1, 1, 0)

# Add anomaly scores for visualization
anomaly_scores = model.decision_function(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Normal", "Anomalous"]
)
cmd.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # Save the plot as an image file
plt.close()

# Visualization: Scatter plot of predictions
plt.figure(figsize=(10, 6))
plt.scatter(
    X_test[:, 0], X_test[:, 1], c=y_pred, cmap="coolwarm", alpha=0.6, edgecolor="k"
)
plt.colorbar(label="Prediction (0: Normal, 1: Anomalous)")
plt.title("Isolation Forest Predictions (Test Set)")
plt.xlabel("Month")
plt.ylabel("Monthly Consumption (kWh)")
plt.grid()
plt.savefig("scatter_plot.png")  # Save the plot as an image file
plt.close()

# Visualization: Anomaly scores for every data point
plt.figure(figsize=(10, 6))
plt.bar(
    range(len(anomaly_scores)), anomaly_scores, color=np.where(y_pred == 1, "r", "b")
)
plt.title("Anomaly Scores for Each Data Point")
plt.xlabel("Data Point Index")
plt.ylabel("Anomaly Score")
plt.axhline(y=0, color="gray", linestyle="--", label="Decision Boundary")
plt.legend()
plt.grid()
plt.savefig("anomaly_scores.png")  # Save the plot as an image file
plt.close()

print("✅ Model training, anomaly detection, and visualization complete.")
