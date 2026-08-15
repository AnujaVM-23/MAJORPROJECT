"""
Step 1: Baseline Logistic Regression model.
Loads the same dataset used by the repo's other models, trains LR only,
saves the fitted model + scaler for reuse by the attack/defense scripts.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DATA_PATH = "datasets/phishyFeatures.csv"
OUT_DIR = "adversarial/artifacts"

import os
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.drop("id", axis=1)

X = df.iloc[:, :-1].values.astype(np.float64)
y = df.iloc[:, -1].values
y = np.where(y == -1, 0, 1)  # map {-1,1} -> {0,1} for consistency with sigmoid/BCE math

feature_names = df.columns[:-1].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(random_state=0, max_iter=1000)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=== Baseline Logistic Regression (clean test set) ===")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall   : {rec*100:.2f}%")
print(f"F1       : {f1*100:.2f}%")
print("Confusion matrix:")
print(cm)

joblib.dump(model, f"{OUT_DIR}/lr_baseline.pkl")
joblib.dump(scaler, f"{OUT_DIR}/scaler.pkl")
np.save(f"{OUT_DIR}/X_train.npy", X_train)
np.save(f"{OUT_DIR}/X_test.npy", X_test)
np.save(f"{OUT_DIR}/y_train.npy", y_train)
np.save(f"{OUT_DIR}/y_test.npy", y_test)
with open(f"{OUT_DIR}/feature_names.txt", "w") as f:
    f.write("\n".join(feature_names))

with open(f"{OUT_DIR}/baseline_metrics.txt", "w") as f:
    f.write(f"Accuracy : {acc*100:.2f}%\n")
    f.write(f"Precision: {prec*100:.2f}%\n")
    f.write(f"Recall   : {rec*100:.2f}%\n")
    f.write(f"F1       : {f1*100:.2f}%\n")
    f.write(f"Confusion matrix:\n{cm}\n")

print(f"\nSaved model, scaler, and splits to {OUT_DIR}/")
