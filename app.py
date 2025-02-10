from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
import os
from deepface import DeepFace

app = Flask(__name__)

# Charger les embeddings pré-enregistrés
EMBEDDINGS_FILE = "authorized_embeddings.json"
with open(EMBEDDINGS_FILE, "r") as f:
    mean_authorized_embeddings = json.load(f)

# Paramètres de la métrique de distance et du seuil
distance_metric = "cosine"
threshold = 0.4

def get_embedding(image_path):
    """
    Extrait l'embedding facial d'une image en utilisant DeepFace.
    """
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        return np.array(embedding[0]["embedding"]).tolist()
    except Exception as e:
        print(f"Erreur lors de l'extraction de l'embedding : {e}")
        return None

def compute_distance(emb1, emb2, metric=distance_metric):
    """
    Calcule la distance entre deux embeddings en utilisant la métrique spécifiée.
    """
    emb1, emb2 = np.array(emb1), np.array(emb2)
    if metric == "cosine":
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return 1 - cos_sim
    elif metric == "euclidean":
        return np.linalg.norm(emb1 - emb2)
    else:
        raise ValueError("Métrique inconnue. Utilisez 'euclidean' ou 'cosine'.")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint pour la prédiction. Reçoit une image et retourne le nom de l'agent reconnu.
    """
    try:
        # Vérifier si un fichier a été envoyé
        if "file" not in request.files:
            return jsonify({"error": "Aucun fichier envoyé"}), 400

        # Sauvegarder l'image temporairement
        file = request.files["file"]
        image_path = "temp.jpg"
        file.save(image_path)

        # Extraire l'embedding de l'image inconnue
        emb_unknown = get_embedding(image_path)
        if emb_unknown is None:
            return jsonify({"error": "Impossible d'extraire l'empreinte faciale"}), 400

        # Comparer avec les embeddings autorisés
        best_agent = "Unknown"
        best_distance = float("inf")

        for agent, mean_emb in mean_authorized_embeddings.items():
            distance = compute_distance(emb_unknown, mean_emb, metric=distance_metric)
            if distance < best_distance:
                best_distance = distance
                best_agent = agent if distance < threshold else "Unknown"

        # Retourner le résultat
        return jsonify({"name": best_agent, "distance": best_distance})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Lancer l'application Flask
    app.run(host="0.0.0.0", port=5000, debug=True)
