import os
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import secrets

# Chargement des variables d'environnement
load_dotenv()

# Configuration sécurisée
class Config:
    MODEL_DIR = Path(os.getenv('MODEL_DIR', '/app/modeles'))
    CONFIG_PATH = Path(os.getenv('CONFIG_PATH', '/app/config/config.yaml'))
    LOG_DIR = Path(os.getenv('LOG_DIR', '/app/logs'))
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max payload
    MODEL_FILE = 'optimus1.joblib'
    
    # Création des répertoires
    for directory in [MODEL_DIR, CONFIG_PATH.parent, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Initialisation Flask
app = Flask(__name__)
app.config.from_object(Config)

# Configuration Swagger sécurisée
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/",
    "securityDefinitions": {
        "APIKeyHeader": {
            "type": "apiKey",
            "name": "X-API-KEY",
            "in": "header"
        }
    }
}

Swagger(app, config=swagger_config, template={
    "info": {
        "title": "API WiFi Optimus",
        "description": "API sécurisée pour la gestion intelligente des réseaux WiFi",
        "contact": {
            "email": "dotapok@gmail.com"
        },
        "version": "1.0.0"
    },
    "schemes": ["https"],
    "security": [{"APIKeyHeader": []}]
})

# Configuration du logging sécurisé
handler = RotatingFileHandler(
    Config.LOG_DIR / 'api.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s [%(ip)s] %(message)s'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

class ModelLoader:
    """Chargeur de modèle sécurisé avec vérification d'intégrité"""
    
    @staticmethod
    def load_model():
        try:
            model_path = Config.MODEL_DIR / Config.MODEL_FILE
            if not model_path.exists():
                raise FileNotFoundError("Fichier modèle introuvable")
                
            # Vérification basique du fichier
            if model_path.stat().st_size < 1024:  # 1KB minimum
                raise ValueError("Fichier modèle corrompu")
                
            return joblib.load(model_path)
        except Exception as e:
            app.logger.error(f"Erreur de chargement du modèle: {str(e)}")
            raise

class APISecurity:
    """Couche de sécurité pour les requêtes API"""
    
    @staticmethod
    def validate_api_key(headers: Dict) -> bool:
        api_key = headers.get('X-API-KEY')
        valid_key = os.getenv('API_KEY')
        return secrets.compare_digest(api_key, valid_key) if valid_key else False
    
    @staticmethod
    def sanitize_input(data: Dict) -> Dict:
        sanitized = {}
        for k, v in data.items():
            if isinstance(v, str):
                sanitized[k] = v.strip()[:100]  # Limite à 100 caractères
            else:
                sanitized[k] = v
        return sanitized

class WiFiPredictor:
    """Gestionnaire principal des prédictions"""
    
    def __init__(self):
        self.pipeline = ModelLoader.load_model()
        self.version = "1.0.0"
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self) -> list:
        """Récupère les noms des features après prétraitement"""
        try:
            return self.pipeline.named_steps['preprocessor'].get_feature_names_out()
        except Exception as e:
            app.logger.error(f"Erreur feature names: {str(e)}")
            return []
    
    def validate_input(self, data: Dict) -> Optional[str]:
        """Valide les données d'entrée"""
        required = {
            'rssi': (float, -90, -30),
            'snr': (float, 0, 40),
            'user_type': (str, ['guest', 'employee', 'vip']),
            'device_os': (str, ['Android', 'iOS', 'Windows', 'macOS']),
            'traffic_upload': (float, 0, 10000),
            'traffic_download': (float, 0, 10000)
        }
        
        for field, (field_type, *constraints) in required.items():
            if field not in data:
                return f"Champ manquant: {field}"
                
            if not isinstance(data[field], field_type):
                return f"Type invalide pour {field}"
                
            if field_type == str and data[field] not in constraints[0]:
                return f"Valeur invalide pour {field}"
                
            if field_type in (float, int) and not (constraints[0] <= data[field] <= constraints[1]):
                return f"Valeur hors limites pour {field}"
        
        return None
    
    def predict(self, data: Dict) -> Dict:
        """Effectue une prédiction sécurisée"""
        try:
            # Validation et nettoyage
            if error := self.validate_input(data):
                raise ValueError(error)
                
            clean_data = APISecurity.sanitize_input(data)
            
            # Conversion en DataFrame
            input_df = pd.DataFrame([clean_data])
            
            # Prédiction
            prediction = self.pipeline.predict(input_df)[0]
            proba = np.max(self.pipeline.predict_proba(input_df))
            
            return {
                "decision": str(prediction),
                "confidence": float(proba),
                "model_version": self.version,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
        except Exception as e:
            app.logger.error(f"Erreur prédiction: {str(e)}")
            raise

# Initialisation
predictor = WiFiPredictor()

# Middleware de sécurité
@app.before_request
def check_auth():
    if request.path == '/health':
        return
        
    if not APISecurity.validate_api_key(request.headers):
        app.logger.warning(f"Accès non autorisé depuis {request.remote_addr}")
        return jsonify({"error": "Accès non autorisé"}), 401

# Endpoints
@app.route('/prediction', methods=['POST'])
@swag_from({
    'tags': ['Prédictions'],
    'security': [{"APIKeyHeader": []}],
    'parameters': [{
        'name': 'body',
        'in': 'body',
        'required': True,
        'schema': {
            '$ref': '#/definitions/DonnesWifi'
        }
    }],
    'responses': {
        200: {
            'description': 'Prédiction réussie',
            'schema': {
                '$ref': '#/definitions/Prediction'
            }
        },
        400: {'description': 'Requête invalide'},
        401: {'description': 'Non autorisé'},
        500: {'description': 'Erreur serveur'}
    }
})
def prediction_endpoint():
    """Endpoint principal de prédiction"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Données JSON requises"}), 400
            
        result = predictor.predict(data)
        app.logger.info(f"Prédiction réussie: {result['decision']}")
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Erreur: {str(e)}")
        return jsonify({"error": "Erreur de traitement"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    return jsonify({
        "status": "healthy",
        "version": predictor.version,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

if __name__ == '__main__':
    ssl_context = None
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        ssl_context = ('cert.pem', 'key.pem')
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        ssl_context=ssl_context,
        debug=os.getenv('DEBUG', 'false').lower() == 'true'
    )