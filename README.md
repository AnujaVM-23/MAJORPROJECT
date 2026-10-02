# Logistic Regression Phishing URL API

`adversarial/main_lr.py` runs a Flask service that checks URLs with a
standardized Logistic Regression model, evaluates a PGD-style adversarial
variant, and then evaluates that variant with an adversarially trained model.

## Requirements

Install the Python dependencies from the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The service expects these files in `adversarial/artifacts/`:

- `lr_baseline.pkl`
- `lr_adversarial_regularized.pkl`
- `scaler.pkl`
- `X_test.npy`
- `y_test.npy`
- Metric JSON files used by `/metrics`

## Run

From the repository root:

```bash
python adversarial/main_lr.py
```

The API starts in Flask debug mode at `http://127.0.0.1:5000`.

## Endpoints

### `POST /predict`

Classifies a live URL and returns its result before the attack, after the
attack, and after the defended model evaluates the attacked features.

Request:

```json
{
  "url": "https://example.com"
}
```

The response includes:

- `input_url`: the submitted URL
- `predictions.before_attack`: baseline prediction
- `predictions.after_attack`: prediction after the generated attack
- `predictions.defence_after_attack`: defended-model prediction
- `predictions.changed_after_attack`: whether the attack changed the label
- `predictions.defence_recovered_prediction`: whether the defense restored the original label
- `attack`: attack name, epsilon, step size, and iteration count

Each prediction contains a `label`, `advice`, `confidence`, and numeric
`class_id`. Class `0` is reported as `Phishy URL`; class `1` is reported as
`Legitimate URL`.

### `GET /metrics`

Returns stored adversarial-evaluation metrics. The response contains baseline
accuracy, attacked accuracy, defended accuracy, robustness gain in percentage
points, model and defense names, and the metric artifact timestamp.

If the metric artifacts are missing, the endpoint returns HTTP `404`. Invalid
JSON returns HTTP `500`.

### `GET /samples`

Returns the predefined held-out sample indices used by the demo.

### `POST /predict_sample`

Runs the same before-attack, after-attack, and defended-model flow for a
held-out sample from `X_test.npy`.

Request:

```json
{
  "index": 1
}
```

The index must be between `0` and the last row in `X_test.npy`. The response
also includes the sample's true label from `y_test.npy`.

## Processing Details

For live URLs, `extract_url_features` creates the feature vector and
`scaler.pkl` standardizes it before prediction. The baseline model is
`lr_baseline.pkl`; the defended model is
`lr_adversarial_regularized.pkl`.

The attack is an untargeted PGD-style update using:

- Epsilon: `0.3`
- Step size (`alpha`): `0.05`
- Iterations: `40`

URL feature extraction may perform network-related checks, so requests can be
slow or fail when DNS, WHOIS, SSL, or the target site is unavailable.
