import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = "adversarial/artifacts"

with open(f"{ART}/pgd_results.json") as f:
    baseline = json.load(f)
with open(f"{ART}/adaptive_attack_results.json") as f:
    defended = json.load(f)

eps_b = [x["epsilon"] for x in baseline["epsilon_sweep"]]
acc_b = [x["adversarial_accuracy"]*100 for x in baseline["epsilon_sweep"]]
eps_d = [x["epsilon"] for x in defended["epsilon_sweep"]]
acc_d = [x["adversarial_accuracy"]*100 for x in defended["epsilon_sweep"]]

plt.figure(figsize=(7,5))
plt.plot(eps_b, acc_b, marker='o', label="Baseline LR (undefended)")
plt.plot(eps_d, acc_d, marker='s', label="Adversarially-trained LR (adaptive attack)")
plt.axhline(y=baseline["clean_accuracy"]*100, linestyle='--', color='gray', alpha=0.6, label="Baseline clean accuracy")
plt.xlabel("PGD epsilon (L-infinity, standardized feature space)")
plt.ylabel("Accuracy (%)")
plt.title("Logistic Regression Robustness under PGD Attack")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{ART}/robustness_curve.png", dpi=150)
print("Saved plot to", f"{ART}/robustness_curve.png")
