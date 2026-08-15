"""
Rigor check: evaluate the adversarially-trained model against an ADAPTIVE
PGD attack (crafted using the defended model's own weights), not just the
static attack that was crafted against the baseline model.

Why this matters: a defense can look artificially strong if it's only
tested against an attack generated for a different (pre-defense) model.
The real test is whether an attacker who has access to the defended model
can still break it.
"""
import numpy as np
import joblib
import json

ART = "adversarial/artifacts"

adv_model = joblib.load(f"{ART}/lr_adversarial.pkl")
scaler = joblib.load(f"{ART}/scaler.pkl")
X_test = np.load(f"{ART}/X_test.npy")
y_test = np.load(f"{ART}/y_test.npy")
X_test_s = scaler.transform(X_test)

w = adv_model.coef_.flatten()
b = adv_model.intercept_[0]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def pgd_attack(X_std, y, w, b, epsilon, alpha, num_iter):
    X_adv = X_std.copy()
    for _ in range(num_iter):
        z = X_adv @ w + b
        p = sigmoid(z)
        grad = np.outer((p - y), w)
        X_adv = X_adv + alpha * np.sign(grad)
        delta = np.clip(X_adv - X_std, -epsilon, epsilon)
        X_adv = X_std + delta
    return X_adv

clean_pred = adv_model.predict(X_test_s)
clean_acc = (clean_pred == y_test).mean()
print(f"Adv-trained model, clean test accuracy: {clean_acc*100:.2f}%\n")

print("Adaptive PGD attack (crafted against the DEFENDED model itself):")
results = []
for epsilon in [0.1, 0.3, 0.5, 1.0]:
    X_adv_s = pgd_attack(X_test_s, y_test, w, b, epsilon=epsilon, alpha=epsilon/6, num_iter=40)
    pred = adv_model.predict(X_adv_s)
    acc = (pred == y_test).mean()
    correctly_classified = clean_pred == y_test
    flipped = (pred != y_test) & correctly_classified
    success_rate = flipped.sum() / correctly_classified.sum()
    print(f"  epsilon={epsilon:>4}: adversarial accuracy = {acc*100:6.2f}%  "
          f"| attack success rate = {success_rate*100:6.2f}%")
    results.append({"epsilon": epsilon, "adversarial_accuracy": acc, "attack_success_rate": success_rate})

with open(f"{ART}/adaptive_attack_results.json", "w") as f:
    json.dump({"clean_accuracy": clean_acc, "epsilon_sweep": results}, f, indent=2)

print(f"\nSaved to {ART}/adaptive_attack_results.json")
