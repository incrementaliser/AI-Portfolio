"""
Base class for time series models
"""
import numpy as np
from typing import Dict, Optional, Any


class BaseTimeSeriesModel:
    """Base wrapper class for time series models to provide unified interface."""
    
    def __init__(self, model: Any, model_type: str):
        """
        Initialize base time series model.
        
        Args:
            model: The underlying model instance
            model_type: Type of model ('statistical', 'ml', 'dl')
        """
        self.model = model
        self.model_type = model_type
        self.is_fitted = False
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """
        Fit the model.
        
        Args:
            y: Target time series
            X: Exogenous variables (optional)
        """
        raise NotImplementedError
    
    def predict(self, steps: int, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            steps: Number of steps to forecast
            X: Exogenous variables for prediction (optional)
            
        Returns:
            Forecasted values
        """
        raise NotImplementedError
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return {}

