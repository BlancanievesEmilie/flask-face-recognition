from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
import os
from deepface import DeepFace
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Active CORS pour autoriser les requêtes depuis ton site web

# Charger les embeddings sauvegardés
EMBEDDINGS_FILE = "authorized_embeddings.json"
if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "r") as f:
        mean_authorized_embeddings = json.load(f)
else:
    mean_authorized_embeddings = {}

distance_metric = "cosine"
threshold = 0.4

def get_embedding(image_path):
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        return np.array(embedding[0]["embedding"]).tolist()
    except:
        return None

def compute_distance(emb1, emb2, metric=distance_metric):
    emb1, emb2 = np.array(emb1), np.array(emb2)
    if metric == "cosine":
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return 1 - cos_sim
    elif metric == "euclidean":
        return np.linalg.norm(emb1 - emb2)
    else:
        raise ValueError("Unknown metric. Use 'euclidean' or 'cosine'.")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Aucun fichier envoyé"}), 400

        file = request.files["file"]
        image_path = "temp.jpg"
        file.save(image_path)

        emb_unknown = get_embedding(image_path)
        if emb_unknown is None:
            return jsonify({"error": "Impossible d'extraire l'empreinte faciale"}), 400

        best_agent = "Unknown"
        best_distance = float("inf")

        for agent, mean_emb in mean_authorized_embeddings.items():
            distance = compute_distance(emb_unknown, mean_emb, metric=distance_metric)
            if distance < best_distance:
                best_distance = distance
                best_agent = agent if distance < threshold else "Unknown"

        return jsonify({"name": best_agent, "distance": best_distance})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
