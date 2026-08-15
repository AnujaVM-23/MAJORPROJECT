# PhishGuard: Logistic Regression under PGD Attack and Adversarial Training

## Dataset
- `datasets/phishyFeatures.csv` from the repo — 11,055 samples, 30 features, all
  categorical-style values in {-1, 0, 1}, label `Result` mapped to {0,1}.
- 75/25 train/test split, stratified, random_state=0.
- Features standardized (`StandardScaler`) before training and attack — required for a
  single epsilon to be meaningful across features.

## 1. Baseline model (Logistic Regression only)
| Metric | Value |
|---|---|
| Accuracy | 93.23% |
| Precision | 93.17% |
| Recall | 94.80% |
| F1 | 93.98% |

Other models in the repo (Random Forest, SVM, etc.) were left in place but excluded from
the active pipeline — this project scope is Logistic Regression only.

## 2. PGD attack on the baseline model
L∞ PGD, 40 iterations, step size = epsilon/6, run in standardized feature space.

| epsilon | Adversarial accuracy | Attack success rate |
|---|---|---|
| 0.1 | 83.18% | 10.79% |
| 0.3 | 48.99% | 47.46% |
| 0.5 | 18.45% | 80.21% |
| 1.0 | 0.40% | 99.57% |

The undefended model degrades sharply — at epsilon=0.3 (a moderate perturbation budget)
nearly half the correctly-classified test points flip to the wrong label.

## 3. Adversarial training defense

**Single-pass adversarial training** (train once on clean + adversarial examples generated
from the baseline model):

| Evaluation | Accuracy |
|---|---|
| Clean test | 89.98% |
| PGD test, **static attack** (crafted against baseline, eps=0.3) | 94.97% |
| PGD test, **adaptive attack** (crafted against the defended model itself, eps=0.3) | 24.67% |

**This is the key finding of the project.** The single-pass defense looks like it
*improves* accuracy under attack (94.97% > clean 89.98%) — but that number is misleading.
It only holds against a *static* attack crafted for the old (undefended) model. Once the
attacker is given access to the defended model's own weights and crafts PGD against it
directly (an **adaptive attack** — the standard way robustness must be evaluated), accuracy
collapses to 24.67%, worse than the undefended baseline's 48.99% at the same epsilon.

**Iterative adversarial training** (regenerate adversarial examples against the *current*
model each round, 5 rounds, Madry-style):

| Round | Clean accuracy | Adaptive-PGD accuracy (eps=0.3) |
|---|---|---|
| 1 | 89.98% | 24.67% |
| 2 | 90.99% | 13.93% |
| 3 | 89.98% | 24.67% |
| 4 | 90.99% | 13.93% |
| 5 | 89.98% | 24.67% |

Training oscillates between two fixed points and never converges to meaningfully better
robustness than round 1.

## Interpretation

Logistic Regression has a **linear decision boundary with a fixed margin**. PGD under an
L∞ budget just needs to push points across that margin — and adversarial training on a
linear model can only shift/rotate that single hyperplane, it cannot locally reshape the
boundary the way a non-linear model (MLP, tree ensemble) can. That's why:
- The defense provides no real robustness once evaluated adaptively.
- Iterative retraining oscillates rather than converging — the model keeps re-fitting to
  whichever adversarial examples were most recently generated, without a way to satisfy
  both the clean and adversarial objectives simultaneously given its limited capacity.

**Takeaway for the report:** this is a legitimate and useful negative result. It
demonstrates (a) that evaluating defenses only against static/pre-generated attacks
overstates robustness — a well-known pitfall in adversarial ML research — and (b) that
adversarial training's effectiveness is capacity-dependent: linear models have a
fundamentally low ceiling for adversarial robustness compared to non-linear classifiers.
This motivates future work (mentioned as such, not required tonight): repeating this
pipeline with the MLP model already explored elsewhere in this project, which has the
non-linear capacity to potentially benefit more from adversarial training.

## Files produced
- `adversarial/train_lr_baseline.py` — baseline LR training
- `adversarial/pgd_attack.py` — PGD attack + epsilon sweep against baseline
- `adversarial/train_lr_adversarial.py` — single-pass adversarial training
- `adversarial/adaptive_attack_check.py` — adaptive-attack rigor check (the key result)
- `adversarial/train_lr_adversarial_iterative.py` — iterative adversarial training
- `adversarial/make_plot.py` — robustness curve plot
- `adversarial/artifacts/` — saved models, scaler, splits, metrics (JSON), and
  `robustness_curve.png`
