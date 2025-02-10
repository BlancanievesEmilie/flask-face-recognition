from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
import os
from deepface import DeepFace
from flask_cors import CORS

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)  # Active CORS pour permettre les requêtes externes

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
    """
    Extrait l'embedding facial d'une image avec DeepFace.
    """
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        return np.array(embedding[0]["embedding"]).tolist()
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction de l'empreinte faciale : {e}")
        return None

def compute_distance(emb1, emb2, metric=distance_metric):
    """
    Calcule la distance entre deux embeddings selon la métrique choisie.
    """
    emb1, emb2 = np.array(emb1), np.array(emb2)
    if metric == "cosine":
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return 1 - cos_sim
    elif metric == "euclidean":
        return np.linalg.norm(emb1 - emb2)
    else:
        raise ValueError("❌ Unknown metric. Use 'euclidean' or 'cosine'.")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Bienvenue sur l'API de reconnaissance faciale ! API en ligne 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    API pour identifier une personne à partir d'une image.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "❌ Aucun fichier envoyé"}), 400

        file = request.files["file"]
        image_path = "temp.jpg"
        file.save(image_path)

        emb_unknown = get_embedding(image_path)
        if emb_unknown is None:
            os.remove(image_path)  # Nettoyage
            return jsonify({"error": "❌ Impossible d'extraire l'empreinte faciale"}), 400

        best_agent = "Unknown"
        best_distance = float("inf")

        for agent, mean_emb in mean_authorized_embeddings.items():
            distance = compute_distance(emb_unknown, mean_emb, metric=distance_metric)
            if distance < best_distance:
                best_distance = distance
                best_agent = agent if distance < threshold else "Unknown"

        os.remove(image_path)  # Nettoyage du fichier temporaire
        return jsonify({"name": best_agent, "distance": best_distance})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render attribue un port dynamique
    print(f"✅ API démarrée sur le port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
