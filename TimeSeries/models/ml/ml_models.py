"""
Machine Learning models for time series forecasting
"""
import numpy as np
from typing import Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb

from models.base import BaseTimeSeriesModel

# ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso


class MLTimeSeriesWrapper(BaseTimeSeriesModel):
    """Wrapper for ML models adapted for time series (with lag features)."""
    
    def __init__(self, model: Any, lookback: int = 10):
        """
        Initialize ML time series model.
        
        Args:
            model: Scikit-learn compatible model
            lookback: Number of lag features to use
        """
        self.lookback = lookback
        self.history = None
        super().__init__(model, 'ml')
    
    def _create_features(self, y: np.ndarray, start_idx: int = 0) -> np.ndarray:
        """
        Create lag features from time series.
        
        Args:
            y: Time series data
            start_idx: Starting index for feature creation
            
        Returns:
            Feature matrix with lag features
        """
        X = []
        for i in range(start_idx, len(y)):
            if i >= self.lookback:
                X.append(y[i-self.lookback:i])
        return np.array(X)
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """
        Fit ML model with lag features.
        
        Args:
            y: Target time series
            X: Additional features (optional, will be concatenated with lag features)
        """
        # Store history for prediction
        self.history = y.copy()
        
        # Create lag features
        X_lags = self._create_features(y)
        y_train = y[self.lookback:]
        
        # Combine with additional features if provided
        if X is not None and len(X) > 0:
            X_additional = X[self.lookback:]
            X_train = np.hstack([X_lags, X_additional])
        else:
            X_train = X_lags
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self
    
    def predict(self, steps: int, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make multi-step predictions.
        
        Args:
            steps: Number of steps to forecast
            X: Additional features for future steps (optional)
            
        Returns:
            Forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        current_history = self.history.copy()
        
        for step in range(steps):
            # Create features from recent history
            X_input = current_history[-self.lookback:].reshape(1, -1)
            
            # Add additional features if provided
            if X is not None and len(X) > step:
                X_additional = X[step].reshape(1, -1)
                X_input = np.hstack([X_input, X_additional])
            
            # Predict next value
            pred = self.model.predict(X_input)[0]
            predictions.append(pred)
            
            # Update history
            current_history = np.append(current_history, pred)
        
        return np.array(predictions)
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        params = self.model.get_params() if hasattr(self.model, 'get_params') else {}
        params['lookback'] = self.lookback
        return params

