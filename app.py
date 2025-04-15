from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load trained model
model = None
model_path = "crop_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
else:
    print(f"⚠️ Model file '{model_path}' not found. Make sure it exists.")

@app.route("/")
def home():
    return render_template("index.html")  # Ensure index.html is in the 'templates' folder

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        features = np.array([[ 
            data["N"], data["P"], data["K"], 
            data["temperature"], data["humidity"], 
            data["ph"], data["rainfall"]
        ]])
        prediction = model.predict(features)[0]
        return jsonify({"crop": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
