"""
Machine Learning models for time series forecasting
"""
import numpy as np
from typing import Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 47
np.random.seed(RANDOM_SEED)

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
    
    def _create_features(self, y: np.ndarray) -> np.ndarray:
        """
        Create lag features from time series.
        
        Args:
            y: Time series data
            
        Returns:
            Feature matrix with lag features
        """
        if len(y) < self.lookback:
            raise ValueError(f"Time series must have at least {self.lookback} values, got {len(y)}")
        
        X = []
        for i in range(self.lookback, len(y)):
            # Features are the lookback values before index i
            X.append(y[i-self.lookback:i])
        return np.array(X)
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """
        Fit ML model with lag features on DIFFERENCED data to handle trends.
        
        Args:
            y: Target time series
            X: Additional features (optional, will be concatenated with lag features)
        """
        # Ensure y is a 1D numpy array
        y = np.asarray(y).flatten()
        if len(y) < self.lookback + 1:
            raise ValueError(f"Time series must have at least {self.lookback + 1} values for training, got {len(y)}")
        
        # Store FULL history (original scale) for prediction
        self.history = y.copy()
        
        # Apply differencing to handle trends
        # y_diff[t] = y[t] - y[t-1]
        self.last_value = y[-1]
        y_diff = np.diff(y)
        
        # Create lag features on DIFFERENCED data
        X_lags = self._create_features(y_diff)
        
        # Targets correspond to difference values
        # y_diff has length len(y)-1
        # X_lags has length len(y_diff) - lookback
        # y_train should be y_diff[lookback:]
        y_train = y_diff[self.lookback:]
        
        # Combine with additional features if provided
        # Note: Handling X with differencing is tricky. For simplicity, we assume X matches original y length
        # and we use X[lookback+1:] to align with y_train
        if X is not None and len(X) > 0:
            X = np.asarray(X)
            if X.ndim == 1:
                # If X is 1D, it should have same length as y
                if len(X) != len(y):
                    raise ValueError(f"Additional features X must have same length as y. Got {len(X)} vs {len(y)}")
                X_additional = X[self.lookback+1:]
            else:
                # If X is 2D, it should have same number of rows as y
                if X.shape[0] != len(y):
                    raise ValueError(f"Additional features X must have same number of rows as y. Got {X.shape[0]} vs {len(y)}")
                X_additional = X[self.lookback+1:]
            
            X_train = np.hstack([X_lags, X_additional])
        else:
            X_train = X_lags
        
        # Ensure feature and target arrays have matching lengths
        if len(X_train) != len(y_train):
            raise ValueError(f"Feature and target arrays must have matching lengths. Got {len(X_train)} vs {len(y_train)}")
        
        # Ensure feature array has correct shape: (n_samples, n_features)
        if X_train.ndim != 2:
            raise ValueError(f"Feature array must be 2D. Got shape {X_train.shape}")
        
        # Ensure we have at least one sample
        if len(X_train) == 0:
            raise ValueError("Cannot fit model with zero training samples")
        
        # Reset model state before fitting (important for refitting)
        if hasattr(self.model, '_Booster'):
            try:
                self.model._Booster = None
            except:
                pass
        
        self.is_fitted = False
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, steps: int, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make multi-step predictions using autoregressive approach on differences.
        
        Args:
            steps: Number of steps to forecast
            X: Additional features for future steps (optional)
            
        Returns:
            Forecasted values (re-integrated to original scale)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.history is None or len(self.history) < self.lookback + 1:
            raise ValueError(f"History must have at least {self.lookback + 1} values for prediction")
        
        predictions_diff = []
        
        # Create differences history
        history_diff = np.diff(self.history)
        current_history_diff = history_diff.copy()
        
        for step in range(steps):
            # Ensure we have enough history
            if len(current_history_diff) < self.lookback:
                raise ValueError(f"Insufficient history for prediction at step {step}")
            
            # Extract the last lookback values as features (from differences)
            X_input = current_history_diff[-self.lookback:].reshape(1, -1)
            
            # Add additional features if provided
            if X is not None and len(X) > step:
                X_additional = X[step]
                if np.ndim(X_additional) == 0:
                    X_additional = np.array([X_additional]).reshape(1, -1)
                else:
                    X_additional = np.atleast_2d(X_additional)
                X_input = np.hstack([X_input, X_additional])
            
            # Predict next difference
            pred_diff = self.model.predict(X_input)[0]
            predictions_diff.append(pred_diff)
            
            # Update difference history
            current_history_diff = np.append(current_history_diff, pred_diff)
        
        # Re-integrate predictions: y_t = y_{t-1} + diff_t
        predictions = []
        last_val = self.history[-1]
        
        for diff in predictions_diff:
            next_val = last_val + diff
            predictions.append(next_val)
            last_val = next_val
        
        return np.array(predictions)
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        params = self.model.get_params() if hasattr(self.model, 'get_params') else {}
        params['lookback'] = self.lookback
        return params

