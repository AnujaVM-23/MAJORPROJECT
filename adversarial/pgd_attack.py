"""
Step 2: PGD (Projected Gradient Descent) evasion attack against the
baseline Logistic Regression model, operating in standardized feature space.

LR decision function: z = w.x + b, p = sigmoid(z)
BCE loss gradient w.r.t. x:  dL/dx = (p - y) * w
We ascend this gradient (sign step) to increase loss, then project back
into the L-infinity ball of radius epsilon around the original x.
"""
import numpy as np
import joblib
import json

ART = "adversarial/artifacts"

model = joblib.load(f"{ART}/lr_baseline.pkl")
scaler = joblib.load(f"{ART}/scaler.pkl")
X_test = np.load(f"{ART}/X_test.npy")
y_test = np.load(f"{ART}/y_test.npy")

w = model.coef_.flatten()  # shape (n_features,)
b = model.intercept_[0]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def pgd_attack(X_std, y, w, b, epsilon, alpha, num_iter):
    """
    X_std: standardized inputs (n_samples, n_features)
    y: true labels {0,1}
    Returns adversarial X_std (still standardized).
    """
    X_adv = X_std.copy()
    for _ in range(num_iter):
        z = X_adv @ w + b
        p = sigmoid(z)
        grad = np.outer((p - y), w)  # dL/dx per sample, shape (n_samples, n_features)
        X_adv = X_adv + alpha * np.sign(grad)
        # project back into L-inf ball of radius epsilon around original point
        delta = np.clip(X_adv - X_std, -epsilon, epsilon)
        X_adv = X_std + delta
    return X_adv


def evaluate(X_std, y, model):
    y_pred = model.predict(X_std)
    acc = (y_pred == y).mean()
    return acc, y_pred


X_test_s = scaler.transform(X_test)
clean_acc, clean_pred = evaluate(X_test_s, y_test, model)
print(f"Clean test accuracy: {clean_acc*100:.2f}%")

results = {"clean_accuracy": clean_acc, "epsilon_sweep": []}

for epsilon in [0.1, 0.3, 0.5, 1.0]:
    X_adv_s = pgd_attack(X_test_s, y_test, w, b, epsilon=epsilon, alpha=epsilon / 6, num_iter=40)
    adv_acc, adv_pred = evaluate(X_adv_s, y_test, model)

    # attack success rate: fraction of originally-correct points that flip
    correctly_classified_mask = clean_pred == y_test
    flipped_mask = (adv_pred != y_test) & correctly_classified_mask
    attack_success_rate = flipped_mask.sum() / correctly_classified_mask.sum()

    print(f"epsilon={epsilon:>4}: adversarial accuracy = {adv_acc*100:6.2f}%  "
          f"| attack success rate = {attack_success_rate*100:6.2f}%")

    results["epsilon_sweep"].append({
        "epsilon": epsilon,
        "adversarial_accuracy": adv_acc,
        "attack_success_rate": attack_success_rate,
    })

# Save the epsilon=0.3 adversarial test set for use in adversarial training evaluation
X_adv_eval = pgd_attack(X_test_s, y_test, w, b, epsilon=0.3, alpha=0.05, num_iter=40)
np.save(f"{ART}/X_test_adv_std.npy", X_adv_eval)

# Generate adversarial *training* examples (single-pass, using baseline model) for defense step
X_train = np.load(f"{ART}/X_train.npy")
y_train = np.load(f"{ART}/y_train.npy")
X_train_s = scaler.transform(X_train)
X_train_adv_s = pgd_attack(X_train_s, y_train, w, b, epsilon=0.3, alpha=0.05, num_iter=40)
np.save(f"{ART}/X_train_adv_std.npy", X_train_adv_s)

# Save a couple of before/after examples for the report
examples = []
for i in range(3):
    z_before = X_test_s[i] @ w + b
    z_after = X_adv_eval[i] @ w + b
    examples.append({
        "true_label": int(y_test[i]),
        "confidence_before": float(sigmoid(z_before)),
        "confidence_after": float(sigmoid(z_after)),
        "pred_before": int(model.predict(X_test_s[i:i+1])[0]),
        "pred_after": int(model.predict(X_adv_eval[i:i+1])[0]),
    })
results["example_flips"] = examples

with open(f"{ART}/pgd_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved PGD results and adversarial datasets to {ART}/")
