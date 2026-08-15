from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS, cross_origin
from feature_extraction.feature_extractor import extract_url_features
app = Flask(__name__)
import joblib
import numpy as np
import json
import os

# Load the data from the pkl file
try:
    with open('phishing.pkl', 'rb') as f:
        data = pickle.load(f)
except FileNotFoundError:
    data = None
cors = CORS(app)


def load_adversarial_metrics():
    metrics_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "adversarial",
        "artifacts",
        "final_summary.json",
    )

    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        summary = json.load(metrics_file)

    baseline_clean = summary["baseline_clean"]["accuracy"]
    baseline_attacked = summary["baseline_under_pgd_eps0.3"]["accuracy"]
    defended_attacked = summary["adv_trained_under_pgd_eps0.3"]["accuracy"]

    return {
        "base_model_accuracy": baseline_clean * 100,
        "after_attack_accuracy": baseline_attacked * 100,
        "defence_accuracy": defended_attacked * 100,
    }

@app.route('/predict', methods=['POST'])
def process_url():
    print("--------------------");
    url = request.json.get('url')
    print(url)
    # Extract features from the new data
    features = extract_url_features(url)
    new_data = np.array(features).reshape(1, -1)

    # Load the trained model
    classifier = joblib.load('trained_models/randomForest.pkl')

    # Predict the class for the new data
    prediction = classifier.predict(new_data)
    data = "abc" 

    # Print the predicted class
    if prediction[0]==1:
        print("Phishy URL")
        data = "Phishy Url\nBe cautious"
    else:
        print(f"Legitimate URL")
        data = "Legitimate Url\nSafe to browse"

    processed_result = {
        "input_url": url,
        "data": data,
        "message": "Processed successfully"
    }
    
    return jsonify(processed_result)


@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        return jsonify(load_adversarial_metrics())
    except FileNotFoundError:
        return jsonify({
            "message": "Metrics file not found. Run adversarial training scripts to generate final_summary.json."
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
    app.run(debug=True)
