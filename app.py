
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load vectorizer and model
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SMS Spam Detection API is running.",
        "available_endpoints": {
            "GET /": "API status check",
            "POST /predict": "Predict whether a message is spam or ham"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400

        if "text" not in data:
            return jsonify({"error": 'Missing required field: "text"'}), 400

        text = str(data["text"]).strip()

        if text == "":
            return jsonify({"error": 'The "text" field cannot be empty.'}), 400

        text_tfidf = tfidf.transform([text])
        prediction = model.predict(text_tfidf)[0]

        probabilities = model.predict_proba(text_tfidf)[0]
        class_probs = {
            model.classes_[i]: float(probabilities[i])
            for i in range(len(model.classes_))
        }

        return jsonify({
            "input_text": text,
            "prediction": prediction,
            "probabilities": class_probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
