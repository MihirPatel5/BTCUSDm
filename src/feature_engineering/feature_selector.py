"""
Feature Selection Module
Selects and ranks the most relevant features for BTCUSD prediction model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureSelector:
    """Selector for identifying the most predictive features"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data/processed/with_features")
        
    def select_features(self, target_column: str = 'target') -> pd.DataFrame:
        """Perform comprehensive feature selection and ranking"""
        
        try:
            # Load data with all features
            data_file = self.data_dir / "BTCUSD_5min_with_time_features.csv"
            if not data_file.exists():
                self.logger.error(f"Feature data file not found: {data_file}")
                return None
            
            data = pd.read_csv(data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.logger.info(f"Performing feature selection on {len(data)} records")
            
            # Define target variable if not already present
            if target_column not in data.columns:
                data = self._define_target_variable(data)
            
            # Separate features and target
            feature_columns = [col for col in data.columns if col not in 
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'data_complete', target_column]]
            
            X = data[feature_columns]
            y = data[target_column]
            
            # Remove columns with too many missing values
            missing_threshold = 0.3  # 30% threshold
            missing_ratios = X.isnull().sum() / len(X)
            cols_to_keep = missing_ratios[missing_ratios <= missing_threshold].index
            X = X[cols_to_keep]
            
            self.logger.info(f"Features after missing value filtering: {len(X.columns)}")
            
            # Handle remaining missing values
            X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove constant features
            constant_features = X.columns[X.std() == 0]
            if len(constant_features) > 0:
                self.logger.info(f"Removing {len(constant_features)} constant features")
                X = X.drop(columns=constant_features)
            
            # Feature selection methods
            results = {}
            
            # 1. Correlation-based selection
            corr_results = self._correlation_filter(X, y)
            results['correlation'] = corr_results
            
            # 2. Univariate statistical tests
            univariate_results = self._univariate_selection(X, y)
            results['univariate'] = univariate_results
            
            # 3. Mutual information
            mi_results = self._mutual_information_selection(X, y)
            results['mutual_info'] = mi_results
            
            # Combine rankings
            combined_ranking = self._combine_rankings(results)
            
            # Select top features
            top_features = self._select_top_features(X, combined_ranking)
            
            # Create final dataset with selected features
            final_data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume', target_column] + top_features].copy()
            
            # Save selected features list
            self._save_feature_rankings(combined_ranking, top_features)
            
            # Save final dataset
            features_dir = Path("data/processed/selected_features")
            features_dir.mkdir(parents=True, exist_ok=True)
            output_file = features_dir / "BTCUSD_5min_selected_features.csv"
            final_data.to_csv(output_file, index=False)
            
            self.logger.info(f"Feature selection completed. Selected {len(top_features)} features.")
            self.logger.info(f"Final dataset saved to: {output_file}")
            
            return final_data
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return None
    
    def _define_target_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """Define target variable for binary classification"""
        
        # Default to 15-minute prediction horizon
        prediction_horizon = self.config.get('data', {}).get('prediction_horizon_minutes', 15)
        periods_ahead = prediction_horizon // 5  # 5-minute intervals
        
        # Binary classification: 1 if price goes up, 0 if down
        data['target'] = (data['close'].shift(-periods_ahead) > data['close']).astype(int)
        
        # Remove last periods_ahead rows as they have NaN targets
        data = data.iloc[:-periods_ahead]
        
        self.logger.info(f"Target variable defined: {data['target'].sum()} up moves, {len(data) - data['target'].sum()} down moves")
        
        return data
    
    def _correlation_filter(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Filter features based on correlation with target and among themselves"""
        
        # Correlation with target
        target_corr = X.apply(lambda col: abs(col.corr(y)))
        
        # Correlation threshold (configurable)
        corr_threshold = 0.01  # Minimum correlation with target
        target_filtered = target_corr[target_corr >= corr_threshold]
        
        self.logger.info(f"Features passing correlation threshold: {len(target_filtered)}")
        
        # Correlation among features (remove highly correlated pairs)
        if len(target_filtered) > 1:
            feature_corr = X[target_filtered.index].corr().abs()
            
            # Remove highly correlated features (threshold from config)
            high_corr_threshold = self.config.get('features', {}).get('correlation_threshold', 0.95)
            
            # Create correlation mask
            upper_triangle = feature_corr.where(
                np.triu(np.ones(feature_corr.shape), k=1).astype(bool)
            )
            
            # Find features with correlation above threshold
            high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > high_corr_threshold)]
            
            # Remove highly correlated features
            filtered_features = target_filtered.drop(high_corr_features, errors='ignore')
            
            self.logger.info(f"Features after correlation filtering: {len(filtered_features)}")
            
            return filtered_features.to_dict()
        
        return target_filtered.to_dict()
    
    def _univariate_selection(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Perform univariate statistical feature selection"""
        
        try:
            # Standardize features for fair comparison
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            
            # Select top k features using ANOVA F-test
            selector = SelectKBest(score_func=f_classif, k=min(50, len(X_scaled.columns)))
            selector.fit(X_scaled, y)
            
            # Get feature scores
            scores = pd.Series(selector.scores_, index=X_scaled.columns)
            scores = scores.sort_values(ascending=False)
            
            # Normalize scores to 0-1 range
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            
            return normalized_scores.to_dict()
            
        except Exception as e:
            self.logger.warning(f"Error in univariate selection: {e}")
            return {}
    
    def _mutual_information_selection(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Perform mutual information feature selection"""
        
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            
            # Calculate mutual information
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
            scores = pd.Series(mi_scores, index=X_scaled.columns)
            scores = scores.sort_values(ascending=False)
            
            # Normalize scores to 0-1 range
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            
            return normalized_scores.to_dict()
            
        except Exception as e:
            self.logger.warning(f"Error in mutual information selection: {e}")
            return {}
    
    def _combine_rankings(self, results: dict) -> pd.Series:
        """Combine different feature rankings into a single ranking"""
        
        # Convert results to DataFrames
        rankings = []
        
        if 'correlation' in results and results['correlation']:
            corr_rank = pd.Series(results['correlation']).rank(ascending=False)
            rankings.append(corr_rank)
        
        if 'univariate' in results and results['univariate']:
            uni_rank = pd.Series(results['univariate']).rank(ascending=False)
            rankings.append(uni_rank)
        
        if 'mutual_info' in results and results['mutual_info']:
            mi_rank = pd.Series(results['mutual_info']).rank(ascending=False)
            rankings.append(mi_rank)
        
        if not rankings:
            self.logger.warning("No feature rankings available for combination")
            return pd.Series()
        
        # Combine rankings (average ranks)
        combined_df = pd.concat(rankings, axis=1)
        combined_df.columns = ['correlation', 'univariate', 'mutual_info'][:len(rankings)]
        
        # Handle missing values
        combined_df = combined_df.fillna(combined_df.max() + 1)  # Assign worst rank to missing
        
        # Average the ranks
        combined_rank = combined_df.mean(axis=1).sort_values()
        
        return combined_rank
    
    def _select_top_features(self, X: pd.DataFrame, ranking: pd.Series, top_k: int = 100) -> list:
        """Select top k features based on combined ranking"""
        
        if ranking.empty:
            self.logger.warning("No feature ranking available, returning all features")
            return list(X.columns)
        
        # Select top features
        top_features = ranking.head(min(top_k, len(ranking))).index.tolist()
        
        self.logger.info(f"Selected top {len(top_features)} features based on combined ranking")
        
        return top_features
    
    def _save_feature_rankings(self, ranking: pd.Series, selected_features: list):
        """Save feature rankings and selected features to files"""
        
        try:
            reports_dir = Path("data/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Save full ranking
            ranking_df = pd.DataFrame({
                'feature': ranking.index,
                'combined_rank': ranking.values,
                'selected': ranking.index.isin(selected_features)
            })
            
            ranking_file = reports_dir / "feature_rankings.csv"
            ranking_df.to_csv(ranking_file, index=False)
            
            # Save selected features list
            selected_file = reports_dir / "selected_features.txt"
            with open(selected_file, 'w') as f:
                for feature in selected_features:
                    f.write(f"{feature}\n")
            
            self.logger.info(f"Feature rankings saved to: {ranking_file}")
            self.logger.info(f"Selected features list saved to: {selected_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving feature rankings: {e}")
    
    def plot_feature_importance(self, ranking: pd.Series, top_k: int = 30):
        """Plot top feature importance"""
        
        try:
            if ranking.empty:
                self.logger.warning("No feature ranking to plot")
                return
            
            # Get top features
            top_features = ranking.head(min(top_k, len(ranking)))
            
            # Create plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x=top_features.values, y=top_features.index)
            plt.title('Top Feature Importance Rankings')
            plt.xlabel('Combined Rank Score')
            plt.ylabel('Features')
            
            # Save plot
            plots_dir = Path("data/reports/plots")
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_file = plots_dir / "feature_importance.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            self.logger.info(f"Feature importance plot saved to: {plot_file}")
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {e}")
