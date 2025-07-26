# Utilise une image Python officielle
FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements d'abord pour profiter du cache Docker
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier l'application
COPY app.py .

# Créer un volume pour les modèles
VOLUME /app/modeles

# Exposer le port de l'API
EXPOSE 5000

# Commande de démarrage
CMD ["python", "app.py"]