"""
Step 3: Adversarial training defense.
Retrain LR on clean + PGD-adversarial training examples (epsilon=0.3),
then compare robustness against the undefended baseline.
"""
import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ART = "adversarial/artifacts"

X_train = np.load(f"{ART}/X_train.npy")
y_train = np.load(f"{ART}/y_train.npy")
X_test = np.load(f"{ART}/X_test.npy")
y_test = np.load(f"{ART}/y_test.npy")
scaler = joblib.load(f"{ART}/scaler.pkl")
baseline_model = joblib.load(f"{ART}/lr_baseline.pkl")

X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
X_train_adv_s = np.load(f"{ART}/X_train_adv_std.npy")
X_test_adv_s = np.load(f"{ART}/X_test_adv_std.npy")

# Augmented training set: clean + adversarial (same true labels - evasion, not poisoning)
X_aug = np.vstack([X_train_s, X_train_adv_s])
y_aug = np.concatenate([y_train, y_train])

adv_model = LogisticRegression(random_state=0, max_iter=1000)
adv_model.fit(X_aug, y_aug)

def metrics(model, X, y, label):
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    f1 = f1_score(y, pred)
    print(f"{label:35s} acc={acc*100:6.2f}%  prec={prec*100:6.2f}%  rec={rec*100:6.2f}%  f1={f1*100:6.2f}%")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

print("=== Summary: Baseline vs Adversarially-Trained LR ===\n")
r1 = metrics(baseline_model, X_test_s,     y_test, "Baseline LR | clean test")
r2 = metrics(baseline_model, X_test_adv_s, y_test, "Baseline LR | PGD-attacked test (eps=0.3)")
r3 = metrics(adv_model,      X_test_s,     y_test, "Adv-trained LR | clean test")
r4 = metrics(adv_model,      X_test_adv_s, y_test, "Adv-trained LR | PGD-attacked test (eps=0.3)")

print(f"\nRobustness gain at eps=0.3: {(r4['accuracy']-r2['accuracy'])*100:.2f} percentage points")

joblib.dump(adv_model, f"{ART}/lr_adversarial.pkl")

summary = {
    "baseline_clean": r1,
    "baseline_under_pgd_eps0.3": r2,
    "adv_trained_clean": r3,
    "adv_trained_under_pgd_eps0.3": r4,
    "robustness_gain_pp": (r4['accuracy']-r2['accuracy'])*100,
}
with open(f"{ART}/final_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved adversarially-trained model and summary to {ART}/")
