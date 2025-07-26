import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG_PATH = 'configuration.yaml'
MODEL_DIR = Path('../modeles')
DATA_DIR = Path('../data')
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

class GenerateurDonnees:
    """Générateur de données réseau optimisé avec des relations complexes"""
    
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        
    def generate_network_data(self, samples=10000):
        """Génère un dataset complet avec des relations non-linéaires"""
        # 1. Paramètres techniques de base
        data = {
            'rssi': self._generate_rssi(samples),
            'snr': self._generate_snr(samples),
            'session_duration': self._generate_session_duration(samples),
            'frequency': self._generate_frequencies(samples),
            'channel': self._generate_channels(samples),
            'latency': self._generate_latency(samples),
            'packet_loss': self._generate_packet_loss(samples),
            'auth_failures': self.rng.poisson(0.7, samples),
            'auth_time': np.clip(self.rng.exponential(2.5, samples), 0.1, 30)
        }
        
        # 2. Contexte temporel doit être généré avant d'être utilisé
        time_data = {
            'hour': self.rng.integers(0, 24, samples),
            'day_of_week': self.rng.integers(0, 7, samples),
            'is_weekend': (self.rng.random(samples) > 0.8).astype(int),
            'is_holiday': (self.rng.random(samples) > 0.95).astype(int)
        }
        data.update(time_data)
        
        # Maintenant on peut calculer is_peak
        data['is_peak'] = (
            ((data['hour'] >= 8) & (data['hour'] <= 10)) |
            ((data['hour'] >= 17) & (data['hour'] <= 20))
        ).astype(int)
        
        # 3. Données utilisateur enrichies
        user_data = {
            'gps_lat': self.rng.uniform(48.81, 48.91, samples),
            'gps_lon': self.rng.uniform(2.25, 2.41, samples),
            'user_type': self.rng.choice(
                ['guest', 'employee', 'vip', 'contractor'], 
                samples, 
                p=[0.55, 0.3, 0.1, 0.05]
            ),
            'device_os': self.rng.choice(
                ['Android', 'iOS', 'Windows', 'macOS', 'Linux'],
                samples,
                p=[0.4, 0.35, 0.15, 0.05, 0.05]
            ),
            'device_age': np.clip(self.rng.normal(2, 1.5, samples), 0, 5)
        }
        data.update(user_data)
        
        # 4. Métriques de trafic dynamiques
        traffic_data = {
            'traffic_upload': self.rng.poisson(200, samples),
            'traffic_download': self.rng.poisson(600, samples),
            'traffic_type': self.rng.choice(
                ['web', 'video', 'voip', 'cloud', 'gaming', 'iot'],
                samples,
                p=[0.4, 0.3, 0.1, 0.1, 0.05, 0.05]
            ),
            'protocol': self.rng.choice(['TCP', 'UDP', 'QUIC'], samples)
        }
        data.update(traffic_data)
        
        df = pd.DataFrame(data)
        
        # Métriques dérivées avancées
        df['traffic_total'] = df['traffic_upload'] + df['traffic_download']
        df['traffic_ratio'] = df['traffic_upload'] / (df['traffic_download'] + 1)
        df['signal_quality'] = df['rssi'] / df['snr']
        df['network_score'] = (
            0.4 * (df['rssi'] / -50) + 
            0.3 * (df['snr'] / 30) + 
            0.2 * (1 - df['packet_loss']) + 
            0.1 * (1 - df['latency'] / 300)
        )
        df['mobility'] = np.clip(self.rng.exponential(0.5, samples), 0, 3)
        
        # Génération de la target avec règles complexes
        df['action'] = self._generate_targets(df)
        
        # Sauvegarde des données
        df.to_csv(DATA_DIR / 'network_data.csv', index=False)
        
        return df
    
    def _generate_rssi(self, samples):
        return np.clip(self.rng.normal(-65, 15, samples), -90, -30)
    
    def _generate_snr(self, samples):
        return np.clip(self.rng.gamma(2.5, 7, samples), 3, 35)
    
    def _generate_session_duration(self, samples):
        return np.clip(self.rng.exponential(1800, samples), 60, 86400)
    
    def _generate_frequencies(self, samples):
        return self.rng.choice([2.4, 5.0], samples, p=[0.6, 0.4])
    
    def _generate_channels(self, samples):
        return np.where(
            self.rng.random(samples) > 0.3,
            self.rng.integers(1, 12),
            self.rng.choice([36, 40, 44, 48, 149, 153, 157, 161])
        )
    
    def _generate_latency(self, samples):
        return np.clip(self.rng.weibull(1.5, samples)*30, 5, 300)
    
    def _generate_packet_loss(self, samples):
        return np.clip(self.rng.beta(1, 20, samples), 0, 0.2)
    
    def _generate_targets(self, df):
        rssi_threshold = -78 if df['frequency'].mean() > 3 else -72
        latency_threshold = 120 if df['is_peak'].mean() > 0.5 else 80
        
        conditions = [
            (df['rssi'] < rssi_threshold) & (df['snr'] < 12),
            (df['latency'] > latency_threshold) & (df['packet_loss'] > 0.03),
            (df['auth_failures'] >= 3) | (df['auth_time'] > 8),
            (df['traffic_total'] > 1200) & (df['user_type'] == 'guest'),
            (df['network_score'] < df['network_score'].quantile(0.2)),
            (df['is_peak'] == 1) & (df['traffic_total'] > 800),
            (df['device_os'] == 'Android') & (df['device_age'] > 3) & (df['packet_loss'] > 0.1)
        ]
        
        choices = [
            'switch_ap',
            'switch_ap',
            'block_user',
            'throttle',
            'load_balance',
            'load_balance',
            'qos_adjust'
        ]
        
        return np.select(conditions, choices, default='no_action')

class EntrainementOptimus:
    """Système d'entraînement optimisé avec évaluation avancée"""
    
    def __init__(self):
        with open(CONFIG_PATH) as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessor = self._build_preprocessor()
        self.model = RandomForestClassifier(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            min_samples_split=self.config.get('min_samples_split', 2),
            class_weight=self.config.get('class_weights'),
            random_state=42,
            n_jobs=-1
        )
    
    def _build_preprocessor(self):
        """Pipeline de prétraitement avancé"""
        numeric_features = [
            'rssi', 'snr', 'session_duration', 'traffic_upload',
            'traffic_download', 'latency', 'packet_loss', 'auth_time',
            'auth_failures', 'device_age', 'traffic_total', 'traffic_ratio',
            'signal_quality', 'network_score'
        ]
        
        categorical_features = [
            'user_type', 'device_os', 'traffic_type', 'protocol', 'frequency'
        ]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
                ('bin', KBinsDiscretizer(n_bins=5, encode='ordinal'), ['channel', 'hour'])
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def train(self):
        """Processus d'entraînement complet avec évaluation détaillée"""
        print("⚡ Début de l'entraînement optimisé")
        
        # 1. Génération des données
        print("🔄 Génération des données avancées...")
        generator = GenerateurDonnees()
        df = generator.generate_network_data(self.config['training_samples'])

        # Vérification de cohérence des classes
        unique_classes = df['action'].unique()
        config_classes = set(self.config.get('class_weights', {}).keys())
        
        if not set(unique_classes).issubset(config_classes):
            missing = set(unique_classes) - config_classes
            raise ValueError(f"Les classes {missing} sont dans les données mais pas dans class_weights")
        
        # 2. Séparation stratifiée
        print("✂️ Séparation des données...")
        X = df.drop('action', axis=1)
        y = df['action']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=42,
            stratify=y
        )
        
        # 3. Pipeline complet
        print("🔧 Construction du pipeline...")
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        
        # 4. Entraînement avec suivi
        print("🏋️ Entraînement en cours...")
        pipeline.fit(X_train, y_train)
        
        # 5. Évaluation approfondie
        print("\n📊 Rapport de performance complet:")
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        print("\n🔢 Matrice de confusion:")
        print(pd.crosstab(y_test, y_pred, rownames=['Réel'], colnames=['Prédit']))
        
        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            print("\n🎯 Importance des features:")
            features = pipeline.named_steps['preprocessor'].get_feature_names_out()
            importances = pipeline.named_steps['classifier'].feature_importances_
            for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1])[:10]:
                print(f"{feat}: {imp:.3f}")
        
        # 6. Sauvegarde
        self._save_model(pipeline)
        
        # 7. Test sur cas critiques
        self._test_edge_cases(pipeline)
    
    def _save_model(self, pipeline):
        """Sauvegarde optimisée avec compression"""
        joblib.dump(
            pipeline, 
            MODEL_DIR / 'optimus1.joblib', 
            compress=('zlib', 3)
        )
        print(f"\n💾 Modèle sauvegardé dans {MODEL_DIR}/optimus1.joblib")
    
    def _test_edge_cases(self, pipeline):
        """Validation sur des cas limites avec toutes les features requises"""
        print("\n🧪 Test sur des cas critiques:")
        
        # Template complet avec toutes les features
        base_case = {
            'rssi': 0, 'snr': 0, 'latency': 0, 'packet_loss': 0,
            'auth_failures': 0, 'traffic_total': 0, 'user_type': 'guest',
            'hour': 12, 'frequency': 2.4, 'device_os': 'Android', 'protocol': 'TCP',
            'network_score': 0, 'device_age': 2.0, 'traffic_type': 'web',
            'traffic_upload': 0, 'signal_quality': 0, 'traffic_download': 0,
            'traffic_ratio': 0, 'auth_time': 0, 'channel': 6, 'session_duration': 1800,
            'is_peak': 0, 'is_weekend': 0, 'is_holiday': 0, 'day_of_week': 1,
            'gps_lat': 48.85, 'gps_lon': 2.35, 'mobility': 1.0
        }

        # Cas 1: Problème réseau sévère
        case1 = base_case.copy()
        case1.update({
            'rssi': -85, 'snr': 8, 'latency': 150, 'packet_loss': 0.1,
            'auth_failures': 4, 'traffic_total': 1500, 'network_score': -0.5,
            'signal_quality': -3.0, 'traffic_upload': 800, 'traffic_download': 700,
            'traffic_ratio': 1.14, 'auth_time': 5.0
        })

        # Cas 2: Utilisateur VIP avec bonnes conditions
        case2 = base_case.copy()
        case2.update({
            'rssi': -50, 'snr': 25, 'latency': 30, 'user_type': 'vip',
            'traffic_total': 200, 'device_os': 'iOS', 'protocol': 'QUIC',
            'hour': 3, 'frequency': 2.4, 'network_score': 0.9
        })

        cases = [case1, case2]
        
        for i, case in enumerate(cases, 1):
            try:
                X_case = pd.DataFrame([case])
                action = pipeline.predict(X_case)[0]
                print(f"Cas {i}: Action recommandée = {action}")
                print(f"Détails: RSSI={case['rssi']}, Latence={case['latency']}, Type={case['user_type']}")
            except Exception as e:
                print(f"Erreur sur le cas {i}: {str(e)}")

if __name__ == '__main__':
    trainer = EntrainementOptimus()
    trainer.train()