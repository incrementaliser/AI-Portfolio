"""
Time series models package

This package contains all time series forecasting models organized by type:
- base: Base class for all models
- statistical: ARIMA, Prophet, ETS, Theta
- ml: Machine learning models with lag features
- neural: Deep learning models via Darts
- factory: Factory for creating models from configuration
"""

# Import base class
from models.base import BaseTimeSeriesModel

# Import statistical models
from models.statistical.statistical import (
    ARIMAWrapper,
    ProphetWrapper,
    ETSWrapper,
    ThetaWrapper
)

# Import ML models
from models.ml.ml_models import MLTimeSeriesWrapper

# Import deep learning models
from models.neural.deep_learning import DartsModelWrapper

# Import factory
from models.factory import TimeSeriesModelFactory

# For backwards compatibility, keep the old import path working
__all__ = [
    'BaseTimeSeriesModel',
    'ARIMAWrapper',
    'ProphetWrapper',
    'ETSWrapper',
    'ThetaWrapper',
    'MLTimeSeriesWrapper',
    'DartsModelWrapper',
    'TimeSeriesModelFactory',
]
