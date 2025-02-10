# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY . .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application va écouter
EXPOSE 5000

# Commande pour lancer l'application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
