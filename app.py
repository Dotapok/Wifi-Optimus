from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, Any

# Configuration initiale
CONFIG = {
    'MODEL_DIR': Path('/app/model'),
    'CONFIG_PATH': Path('/app/config/config.yaml'),
    'LOG_DIR': Path('/app/logs')
}

# Création des répertoires nécessaires
for directory in CONFIG.values():
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['SWAGGER'] = {
    'title': 'API WiFi Optimus - Système Intelligent de Gestion Réseau',
    'uiversion': 3,
    'specs_route': '/docs/',
    'description': """
    API d'inférence pour le système intelligent de gestion des réseaux WiFi.
    Ce système utilise un modèle ML pour optimiser:
    - La répartition des charges
    - La sécurité du réseau
    - L'expérience utilisateur
    """
}

Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "API WiFi Optimus",
        "version": "1.0",
        "contact": {
            "email": "support@wifioptimus.com"
        }
    },
    "consumes": [
        "application/json"
    ],
    "produces": [
        "application/json"
    ],
    "definitions": {
        "DonnesWifi": {
            "type": "object",
            "required": ["rssi", "snr", "user_type"],
            "properties": {
                "rssi": {"type": "number", "example": -68.5},
                "snr": {"type": "number", "example": 22.3},
                "session_duration": {"type": "integer", "example": 300},
                "frequency": {"type": "number", "example": 5.0},
                "channel": {"type": "integer", "example": 36},
                "user_type": {
                    "type": "string", 
                    "enum": ["guest", "employee", "vip"],
                    "example": "guest"
                },
                "device_os": {
                    "type": "string",
                    "enum": ["Android", "iOS", "Windows", "macOS"],
                    "example": "Android"
                },
                "traffic_upload": {"type": "number", "example": 150.2},
                "traffic_download": {"type": "number", "example": 450.7},
                "latency": {"type": "number", "example": 45.1},
                "packet_loss": {"type": "number", "example": 0.02}
            }
        },
        "Prediction": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["switch_ap", "block_user", "load_balance", 
                            "change_channel", "throttle", "no_action"],
                    "example": "switch_ap"
                },
                "confidence": {"type": "number", "example": 0.92},
                "model_version": {"type": "string", "example": "1.0.0"},
                "timestamp": {"type": "string", "format": "date-time"}
            }
        }
    }
})

class WiFiAIModel:
    """Classe de gestion du modèle WiFi AI"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.version = "1.0.0"
        self.load_model()

    def load_model(self):
        """Charge le modèle et le scaler depuis les fichiers persistants"""
        try:
            model_path = CONFIG['MODEL_DIR'] / 'wifi_ai_pipeline.joblib'
            pipeline = joblib.load(model_path)
            
            self.model = pipeline.named_steps['classifier']
            self.scaler = pipeline.named_steps['preprocessor']
            
            app.logger.info(f"Modèle chargé (version {self.version})")
        except Exception as e:
            app.logger.error(f"Erreur de chargement du modèle: {str(e)}")
            raise RuntimeError("Échec du chargement du modèle")

    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Transforme les données d'entrée en DataFrame formaté"""
        required_fields = {
            'rssi': float,
            'snr': float,
            'user_type': str,
            'device_os': str,
            'traffic_upload': float,
            'traffic_download': float
        }
        
        # Validation des champs obligatoires
        for field, field_type in required_fields.items():
            if field not in input_data:
                raise ValueError(f"Champ manquant: {field}")
            if not isinstance(input_data[field], field_type):
                raise ValueError(f"Type invalide pour {field}")
        
        return pd.DataFrame([input_data])

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Effectue une prédiction sur les données d'entrée"""
        try:
            # Préparation des données
            input_df = self.preprocess_input(input_data)
            
            # Transformation
            input_processed = self.scaler.transform(input_df)
            
            # Prédiction
            prediction = self.model.predict(input_processed)[0]
            confidence = np.max(self.model.predict_proba(input_processed))
            
            return {
                "decision": prediction,
                "confidence": float(confidence),
                "model_version": self.version,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            app.logger.error(f"Erreur de prédiction: {str(e)}")
            raise RuntimeError("Échec de la prédiction")

# Initialisation du modèle
wifi_model = WiFiAIModel()

@app.route('/prediction', methods=['POST'])
def handle_prediction():
    """
    Endpoint principal de prédiction
    ---
    tags:
      - Prédictions
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Aucune donnée fournie"}), 400
            
        result = wifi_model.predict(data)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        app.logger.exception("Erreur inattendue")
        return jsonify({"error": "Erreur interne du serveur"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de vérification de santé
    ---
    tags:
      - Administration
    responses:
      200:
        description: Statut du service
    """
    return jsonify({
        "status": "healthy",
        "model_version": wifi_model.version,
        "service": "wifi-optimus-api"
    })

if __name__ == '__main__':
    import logging
    logging.basicConfig(
        filename=CONFIG['LOG_DIR'] / 'api.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False  # Désactivé en production
    )