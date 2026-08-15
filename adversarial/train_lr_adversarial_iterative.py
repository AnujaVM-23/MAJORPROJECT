"""
Iterative adversarial training: unlike the single-pass version, this
regenerates PGD adversarial examples against the CURRENT model at each
round, then retrains. This is closer to standard adversarial training
practice (e.g. Madry et al.) and should give more genuine robustness
than the single-pass version, which only ever saw attacks crafted
against the original baseline.
"""
import numpy as np
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ART = "adversarial/artifacts"
EPSILON = 0.3
ALPHA = 0.05
PGD_ITERS = 40
ROUNDS = 5

X_train = np.load(f"{ART}/X_train.npy")
y_train = np.load(f"{ART}/y_train.npy")
X_test = np.load(f"{ART}/X_test.npy")
y_test = np.load(f"{ART}/y_test.npy")
scaler = joblib.load(f"{ART}/scaler.pkl")

X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)


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


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
    }


model = LogisticRegression(random_state=0, max_iter=1000)
model.fit(X_train_s, y_train)  # round 0 = plain baseline, to seed the loop

history = []
for round_num in range(1, ROUNDS + 1):
    w = model.coef_.flatten()
    b = model.intercept_[0]

    # generate adversarial train examples against the CURRENT model
    X_train_adv_s = pgd_attack(X_train_s, y_train, w, b, EPSILON, ALPHA, PGD_ITERS)
    X_aug = np.vstack([X_train_s, X_train_adv_s])
    y_aug = np.concatenate([y_train, y_train])

    model = LogisticRegression(random_state=0, max_iter=1000)
    model.fit(X_aug, y_aug)

    # evaluate: clean + adaptive attack against THIS round's model
    w = model.coef_.flatten()
    b = model.intercept_[0]
    X_test_adv_s = pgd_attack(X_test_s, y_test, w, b, EPSILON, ALPHA, PGD_ITERS)

    clean_metrics = evaluate(model, X_test_s, y_test)
    adv_metrics = evaluate(model, X_test_adv_s, y_test)

    print(f"Round {round_num}: clean acc={clean_metrics['accuracy']*100:6.2f}%  "
          f"| adaptive-PGD acc={adv_metrics['accuracy']*100:6.2f}%")

    history.append({
        "round": round_num,
        "clean": clean_metrics,
        "adaptive_pgd_eps0.3": adv_metrics,
    })

joblib.dump(model, f"{ART}/lr_adversarial_iterative.pkl")
with open(f"{ART}/iterative_adv_training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nFinal round {ROUNDS} clean accuracy:        {history[-1]['clean']['accuracy']*100:.2f}%")
print(f"Final round {ROUNDS} adaptive-PGD accuracy: {history[-1]['adaptive_pgd_eps0.3']['accuracy']*100:.2f}%")
print(f"\nSaved iteratively adversarially-trained model and history to {ART}/")
