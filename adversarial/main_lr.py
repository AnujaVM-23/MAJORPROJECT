"""
Live URL phishing checker backend - uses the adversarially-studied
Logistic Regression model (adversarial/artifacts/lr_baseline.pkl) instead
of the original Random Forest model.

Run this instead of main.py to power the existing React frontend with
the new LR model. Same port (5000), same /predict endpoint - the
frontend needs no changes.
"""
import sys
import os
import json
from datetime import datetime, timezone

# Make this runnable from anywhere: add the repo root (parent of this
# script's folder) to sys.path so `feature_extraction` can be imported,
# regardless of the current working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from flask import Flask, request, jsonify
from flask_cors import CORS
from feature_extraction.feature_extractor import extract_url_features
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

ARTIFACTS = os.path.join(REPO_ROOT, 'adversarial', 'artifacts')

# Load the LR model + the scaler it was trained with (LR needs standardized
# features - unlike the Random Forest model, which didn't require scaling)
model = joblib.load(os.path.join(ARTIFACTS, 'lr_baseline.pkl'))
scaler = joblib.load(os.path.join(ARTIFACTS, 'scaler.pkl'))


def load_adversarial_metrics():
    summary_path = os.path.join(ARTIFACTS, 'final_summary.json')

    with open(summary_path, 'r', encoding='utf-8') as summary_file:
        summary = json.load(summary_file)

    base_model_accuracy = summary["baseline_clean"]["accuracy"] * 100
    after_attack_accuracy = summary["baseline_under_pgd_eps0.3"]["accuracy"] * 100
    defence_accuracy = summary["adv_trained_under_pgd_eps0.3"]["accuracy"] * 100
    robustness_gain_pp = defence_accuracy - after_attack_accuracy

    return {
        "base_model_accuracy": base_model_accuracy,
        "after_attack_accuracy": after_attack_accuracy,
        "defence_accuracy": defence_accuracy,
        "robustness_gain_pp": robustness_gain_pp,
        "model_name": "Logistic Regression",
        "defense_name": "Adversarial Training",
        "metrics_updated_at": datetime.fromtimestamp(
            os.path.getmtime(summary_path),
            tz=timezone.utc
        ).isoformat(),
    }


@app.route('/predict', methods=['POST'])
def process_url():
    url = request.json.get('url')
    print(f"Checking URL: {url}")

    features = extract_url_features(url)
    raw = np.array(features).reshape(1, -1)
    scaled = scaler.transform(raw)

    prediction = model.predict(scaled)[0]
    confidence = model.predict_proba(scaled)[0][int(prediction)]

    if prediction == 1:
        data = f"Phishy URL\nBe cautious (confidence: {confidence*100:.1f}%)"
    else:
        data = f"Legitimate URL\nSafe to browse (confidence: {confidence*100:.1f}%)"

    return jsonify({
        "input_url": url,
        "data": data,
        "message": "Processed successfully",
        "model": "Logistic Regression"
    })


@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        return jsonify(load_adversarial_metrics())
    except FileNotFoundError:
        return jsonify({
            "message": "Metrics file not found. Generate adversarial/artifacts/final_summary.json first."
        }), 404
    except KeyError as err:
        return jsonify({
            "message": f"Metrics file is missing expected key: {err}"
        }), 500
    except json.JSONDecodeError:
        return jsonify({
            "message": "Metrics file is invalid JSON."
        }), 500


if __name__ == '__main__':
    print("Serving predictions with the Logistic Regression model.")
    print("Endpoint: http://127.0.0.1:5000/predict")
    app.run(debug=True)
