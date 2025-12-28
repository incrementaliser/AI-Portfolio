"""
Deep Learning models for time series forecasting (via Darts)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from models.base import BaseTimeSeriesModel

# Darts models for deep learning
from darts import TimeSeries
from darts.models import (
    RNNModel, 
    NBEATSModel,
    TransformerModel,
    TFTModel,
    TCNModel
)

class DartsModelWrapper(BaseTimeSeriesModel):
    """Wrapper for Darts deep learning models."""
    
    def __init__(self, model_class, input_chunk_length: int, output_chunk_length: int, **kwargs):
        """
        Initialize Darts model.
        
        Args:
            model_class: Darts model class
            input_chunk_length: Length of input sequences
            output_chunk_length: Length of output sequences
            **kwargs: Model-specific parameters
        """
        self.model_class = model_class
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.model_params = kwargs
        
        model = model_class(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            **kwargs
        )
        super().__init__(model, 'dl')
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None, epochs: int = 100):
        """
        Fit Darts model.
        
        Args:
            y: Target series with datetime index
            X: Covariates (optional)
            epochs: Number of training epochs
        """
        # Convert to Darts TimeSeries
        series = TimeSeries.from_series(y)
        
        # Fit model
        self.model.fit(series, epochs=epochs, verbose=False)
        self.is_fitted = True
        return self
    
    def predict(self, steps: int, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions with Darts model.
        
        Args:
            steps: Number of steps to forecast
            X: Future covariates (optional)
            
        Returns:
            Forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.model.predict(n=steps)
        return forecast.values().flatten()
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'input_chunk_length': self.input_chunk_length,
            'output_chunk_length': self.output_chunk_length,
            **self.model_params
        }


