import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class GeopoliticalMLModels:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.history = {}
        
    def build_volatility_predictor(self, X_train, y_train, X_test, y_test):
        """Random Forest for volatility/shock prediction"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            min_samples_split=5
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['volatility_rf'] = model
        self.history['volatility_rf'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        
        return model, {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': y_pred}
    
    def build_market_shock_detector(self, X_train, y_train, X_test, y_test):
        """Gradient Boosting for market shock prediction"""
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['market_shock_gb'] = model
        self.history['market_shock_gb'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        
        return model, {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': y_pred}
    
    def build_time_series_forecaster(self, X_train, y_train, X_test, y_test):
        """Neural Network for time series forecasting"""
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['timeseries_mlp'] = model
        self.history['timeseries_mlp'] = {'rmse': rmse, 'mae': mae, 'r2': r2}
        
        return model, {'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': y_pred}
    
    def build_anomaly_detector(self, X_train):
        """Isolation Forest for anomaly detection"""
        model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        model.fit(X_train)
        self.models['anomaly_if'] = model
        
        return model
    
    def detect_geopolitical_anomalies(self, data, columns=None):
        """Detect anomalies in geopolitical metrics"""
        if columns is None:
            columns = [col for col in data.columns if '_Shock' in col]
        
        X = data[columns].fillna(0)
        model = self.models.get('anomaly_if')
        
        if model is None:
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(X)
            self.models['anomaly_if'] = model
        
        predictions = model.predict(X)
        anomaly_scores = model.score_samples(X)
        
        anomalies = data[predictions == -1]
        
        return {
            'anomalies_count': len(anomalies),
            'anomaly_dates': anomalies.index.tolist(),
            'anomaly_scores': anomaly_scores,
            'anomalies': anomalies
        }
    
    def feature_importance_analysis(self, model_name, feature_names):
        """Analyze feature importance"""
        model = self.models.get(model_name)
        
        if model is None or not hasattr(model, 'feature_importances_'):
            return None
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10
        
        return {
            'features': [feature_names[i] for i in indices],
            'importance_scores': [importances[i] for i in indices]
        }
    
    def save_models(self, path_prefix='models/'):
        """Save trained models"""
        for name, model in self.models.items():
            joblib.dump(model, f'{path_prefix}{name}.pkl')
        print(f"✅ Models saved to {path_prefix}")
    
    def load_models(self, path_prefix='models/'):
        """Load trained models"""
        for name in self.models.keys():
            self.models[name] = joblib.load(f'{path_prefix}{name}.pkl')
        print(f"✅ Models loaded from {path_prefix}")
    
    def get_model_metrics(self):
        """Get all model metrics"""
        return self.history
    
    def predict(self, model_name, X):
        """Make predictions with specific model"""
        model = self.models.get(model_name)
        if model is None:
            return None
        return model.predict(X)