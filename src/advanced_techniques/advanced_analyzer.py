"""
Advanced Techniques Module
Implements advanced modeling techniques for BTCUSD prediction models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import advanced modeling libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import xgboost as xgb
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
    from tensorflow.keras.optimizers import Adam
    import shap
    HAS_ADVANCED_LIBS = True
except ImportError as e:
    print(f"Warning: Some advanced libraries not available: {e}")
    HAS_ADVANCED_LIBS = False


class AdvancedAnalyzer:
    """Advanced techniques for BTCUSD prediction models"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data")
        self.models_dir = Path("models")
        self.advanced_dir = Path("advanced_techniques")
        self.advanced_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_ADVANCED_LIBS:
            self.logger.warning("Some advanced libraries not available. Advanced features will be limited.")
    
    def implement_ensemble_methods(self, data_file: str) -> dict:
        """Implement advanced ensemble methods"""
        
        if not HAS_ADVANCED_LIBS:
            self.logger.warning("Advanced libraries not available. Skipping ensemble methods.")
            return {}
        
        try:
            self.logger.info("Implementing advanced ensemble methods")
            
            # Load data
            data_path = Path(data_file)
            if not data_path.exists():
                self.logger.error(f"Data file not found: {data_path}")
                return {}
            
            data = pd.read_csv(data_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                               'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
            
            X = data[feature_columns]
            y = data['target_binary']
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Split data
            split_ratio = self.config.get('model', {}).get('train_test_split', 0.8)
            split_index = int(len(data) * split_ratio)
            
            X_train = X.iloc[:split_index]
            y_train = y.iloc[:split_index]
            X_test = X.iloc[split_index:]
            y_test = y.iloc[split_index:]
            
            # Advanced ensemble models
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42),
                'svm': SVC(probability=True, random_state=42)
            }
            
            results = {}
            trained_models = {}
            
            # Train models
            for name, model in models.items():
                try:
                    self.logger.info(f"Training {name} model")
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                    
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall': recall_score(y_test, y_pred, zero_division=0),
                        'f1_score': f1_score(y_test, y_pred, zero_division=0),
                        'roc_auc': roc_auc_score(y_test, y_pred_proba)
                    }
                    
                    results[name] = metrics
                    trained_models[name] = model
                    
                    self.logger.info(f"{name} results: {metrics}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {name}: {e}")
                    results[name] = {'error': str(e)}
            
            # Save trained models
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for name, model in trained_models.items():
                model_path = self.models_dir / f"{name}_model_{timestamp}.pkl"
                joblib.dump(model, model_path)
                self.logger.info(f"{name} model saved to: {model_path}")
            
            # Save results
            results_file = self.advanced_dir / f"ensemble_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Ensemble methods completed. Results saved to: {results_file}")
            
            return {
                'timestamp': timestamp,
                'results': results,
                'models_saved': [str(self.models_dir / f"{name}_model_{timestamp}.pkl") for name in trained_models.keys()]
            }
            
        except Exception as e:
            self.logger.error(f"Error implementing ensemble methods: {e}")
            return {}
    
    def implement_regime_detection(self, data_file: str) -> dict:
        """Implement market regime detection using clustering"""
        
        if not HAS_ADVANCED_LIBS:
            self.logger.warning("Advanced libraries not available. Skipping regime detection.")
            return {}
        
        try:
            self.logger.info("Implementing market regime detection")
            
            # Load data
            data_path = Path(data_file)
            if not data_path.exists():
                self.logger.error(f"Data file not found: {data_path}")
                return {}
            
            data = pd.read_csv(data_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Select volatility and trend features for regime detection
            regime_features = [
                'volatility_atr_14', 'volatility_bbands_width_20',
                'trend_sma_50', 'trend_ema_20', 'trend_macd',
                'volume_sma_50', 'rsi_14'
            ]
            
            # Filter available features
            available_features = [f for f in regime_features if f in data.columns]
            
            if len(available_features) < 3:
                self.logger.warning("Insufficient features for regime detection")
                return {}
            
            X_regime = data[available_features].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_regime)
            
            # Apply clustering to detect regimes
            n_regimes = self.config.get('advanced', {}).get('n_regimes', 3)
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            regime_labels = kmeans.fit_predict(X_scaled)
            
            # Add regime labels to data
            data['regime'] = regime_labels
            
            # Calculate regime statistics
            regime_stats = {}
            for regime in range(n_regimes):
                regime_data = data[data['regime'] == regime]
                regime_stats[f'regime_{regime}'] = {
                    'count': len(regime_data),
                    'percentage': len(regime_data) / len(data) * 100,
                    'avg_volatility': regime_data['volatility_atr_14'].mean() if 'volatility_atr_14' in regime_data.columns else 0,
                    'avg_trend': regime_data['trend_sma_50'].mean() if 'trend_sma_50' in regime_data.columns else 0,
                    'avg_rsi': regime_data['rsi_14'].mean() if 'rsi_14' in regime_data.columns else 0
                }
            
            # Save regime detection results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save data with regime labels
            regime_data_file = self.advanced_dir / f"BTCUSD_with_regimes_{timestamp}.csv"
            data.to_csv(regime_data_file, index=False)
            
            # Save regime statistics
            stats_file = self.advanced_dir / f"regime_stats_{timestamp}.json"
            with open(stats_file, 'w') as f:
                json.dump(regime_stats, f, indent=2, default=str)
            
            # Save model
            model_file = self.models_dir / f"regime_detection_model_{timestamp}.pkl"
            joblib.dump({
                'kmeans': kmeans,
                'scaler': scaler,
                'features': available_features
            }, model_file)
            
            self.logger.info(f"Regime detection completed. Results saved to: {stats_file}")
            
            return {
                'timestamp': timestamp,
                'regime_stats': regime_stats,
                'n_regimes': n_regimes,
                'files_saved': {
                    'regime_data': str(regime_data_file),
                    'regime_stats': str(stats_file),
                    'model': str(model_file)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error implementing regime detection: {e}")
            return {}
    
    def implement_feature_importance_analysis(self, data_file: str) -> dict:
        """Implement advanced feature importance analysis using SHAP"""
        
        if not HAS_ADVANCED_LIBS:
            self.logger.warning("Advanced libraries not available. Skipping feature importance analysis.")
            return {}
        
        try:
            self.logger.info("Implementing feature importance analysis")
            
            # Load data
            data_path = Path(data_file)
            if not data_path.exists():
                self.logger.error(f"Data file not found: {data_path}")
                return {}
            
            data = pd.read_csv(data_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Prepare features and target
            feature_columns = [col for col in data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                               'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
            
            X = data[feature_columns]
            y = data['target_binary']
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Split data
            split_ratio = self.config.get('model', {}).get('train_test_split', 0.8)
            split_index = int(len(data) * split_ratio)
            
            X_train = X.iloc[:split_index]
            y_train = y.iloc[:split_index]
            X_test = X.iloc[split_index:]
            y_test = y.iloc[split_index:]
            
            # Train a model for SHAP analysis
            model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # SHAP analysis
            self.logger.info("Calculating SHAP values")
            explainer = shap.TreeExplainer(model)
            
            # Use subset for SHAP calculation (for performance)
            sample_size = min(1000, len(X_train))
            X_sample = X_train.sample(n=sample_size, random_state=42)
            
            shap_values = explainer.shap_values(X_sample)
            
            # Calculate mean absolute SHAP values
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Save feature importance results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save importance dataframe
            importance_file = self.advanced_dir / f"feature_importance_{timestamp}.csv"
            importance_df.to_csv(importance_file, index=False)
            
            # Save top features
            top_features = importance_df.head(20).to_dict('records')
            
            results = {
                'timestamp': timestamp,
                'top_features': top_features,
                'total_features': len(importance_df),
                'files_saved': {
                    'importance_csv': str(importance_file)
                }
            }
            
            # Save results
            results_file = self.advanced_dir / f"feature_importance_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Feature importance analysis completed. Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error implementing feature importance analysis: {e}")
            return {}
    
    def implement_dimensionality_reduction(self, data_file: str) -> dict:
        """Implement dimensionality reduction techniques"""
        
        if not HAS_ADVANCED_LIBS:
            self.logger.warning("Advanced libraries not available. Skipping dimensionality reduction.")
            return {}
        
        try:
            self.logger.info("Implementing dimensionality reduction")
            
            # Load data
            data_path = Path(data_file)
            if not data_path.exists():
                self.logger.error(f"Data file not found: {data_path}")
                return {}
            
            data = pd.read_csv(data_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Prepare features
            feature_columns = [col for col in data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                               'target_binary', 'target_regression', 'target_multiclass', 'target_multiclass_numeric']]
            
            X = data[feature_columns]
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA
            self.logger.info("Applying PCA")
            n_components = min(50, len(feature_columns), len(X_scaled))
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Apply t-SNE (on PCA components for performance)
            self.logger.info("Applying t-SNE")
            tsne_components = min(10, n_components)
            X_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_pca[:, :tsne_components])
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save PCA results
            pca_results = pd.DataFrame({
                'component': range(1, len(explained_variance) + 1),
                'explained_variance': explained_variance,
                'cumulative_variance': cumulative_variance
            })
            
            pca_file = self.advanced_dir / f"pca_results_{timestamp}.csv"
            pca_results.to_csv(pca_file, index=False)
            
            # Save t-SNE results
            tsne_df = pd.DataFrame({
                'tsne_1': X_tsne[:, 0],
                'tsne_2': X_tsne[:, 1]
            })
            
            tsne_file = self.advanced_dir / f"tsne_results_{timestamp}.csv"
            tsne_df.to_csv(tsne_file, index=False)
            
            # Save models
            pca_model_file = self.models_dir / f"pca_model_{timestamp}.pkl"
            joblib.dump({
                'pca': pca,
                'scaler': scaler
            }, pca_model_file)
            
            results = {
                'timestamp': timestamp,
                'pca_components': n_components,
                'explained_variance_ratio': float(np.sum(explained_variance)),
                'cumulative_variance_95': int(np.argmax(cumulative_variance >= 0.95) + 1) if np.any(cumulative_variance >= 0.95) else n_components,
                'files_saved': {
                    'pca_csv': str(pca_file),
                    'tsne_csv': str(tsne_file),
                    'pca_model': str(pca_model_file)
                }
            }
            
            # Save results
            results_file = self.advanced_dir / f"dimensionality_reduction_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Dimensionality reduction completed. Results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error implementing dimensionality reduction: {e}")
            return {}
    
    def run_advanced_analysis_cycle(self) -> dict:
        """Run a complete advanced analysis cycle"""
        
        try:
            self.logger.info("Starting advanced analysis cycle")
            
            # Get latest data
            latest_data_file = self.data_dir / "processed" / "BTCUSD_5min_processed.csv"
            if not latest_data_file.exists():
                self.logger.error(f"Latest data file not found: {latest_data_file}")
                return {'status': 'failed', 'reason': 'Latest data file not found'}
            
            # Run all advanced techniques
            results = {
                'timestamp': datetime.now().isoformat(),
                'ensemble_methods': {},
                'regime_detection': {},
                'feature_importance': {},
                'dimensionality_reduction': {}
            }
            
            # Run ensemble methods
            results['ensemble_methods'] = self.implement_ensemble_methods(str(latest_data_file))
            
            # Run regime detection
            results['regime_detection'] = self.implement_regime_detection(str(latest_data_file))
            
            # Run feature importance analysis
            results['feature_importance'] = self.implement_feature_importance_analysis(str(latest_data_file))
            
            # Run dimensionality reduction
            results['dimensionality_reduction'] = self.implement_dimensionality_reduction(str(latest_data_file))
            
            # Save overall results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = self.advanced_dir / f"advanced_analysis_cycle_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Advanced analysis cycle completed. Results saved to: {results_file}")
            
            return {
                'status': 'completed',
                'timestamp': timestamp,
                'results_file': str(results_file),
                'techniques_applied': [
                    'ensemble_methods' if results['ensemble_methods'] else None,
                    'regime_detection' if results['regime_detection'] else None,
                    'feature_importance' if results['feature_importance'] else None,
                    'dimensionality_reduction' if results['dimensionality_reduction'] else None
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in advanced analysis cycle: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def get_advanced_techniques_summary(self) -> dict:
        """Get summary of advanced techniques capabilities"""
        
        try:
            summary = {
                'has_advanced_libraries': HAS_ADVANCED_LIBS,
                'techniques_available': [
                    'ensemble_methods',
                    'regime_detection',
                    'feature_importance_analysis',
                    'dimensionality_reduction'
                ] if HAS_ADVANCED_LIBS else [],
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting advanced techniques summary: {e}")
            return {}
