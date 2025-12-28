"""
Statistical time series models (ARIMA, Prophet, ETS, Theta)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from models.base import BaseTimeSeriesModel

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from prophet import Prophet


class ARIMAWrapper(BaseTimeSeriesModel):
    """Wrapper for ARIMA model."""
    
    def __init__(self, order: tuple = (1, 1, 1)):
        """
        Initialize ARIMA model.
        
        Args:
            order: Tuple of (p, d, q) for ARIMA
        """
        self.order = order
        super().__init__(None, 'statistical')
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """Fit ARIMA model."""
        self.model = ARIMA(y, order=self.order, exog=X)
        self.fitted_model = self.model.fit()
        self.is_fitted = True
        return self
    
    def predict(self, steps: int, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Make ARIMA predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        forecast = self.fitted_model.forecast(steps=steps, exog=X)
        return np.array(forecast)
    
    def get_params(self) -> Dict:
        """Get ARIMA parameters."""
        return {'order': self.order}


class ProphetWrapper(BaseTimeSeriesModel):
    """Wrapper for Facebook Prophet model."""
    
    def __init__(self, **kwargs):
        """
        Initialize Prophet model.
        
        Args:
            **kwargs: Prophet parameters
        """
        self.prophet_params = kwargs
        super().__init__(Prophet(**kwargs), 'statistical')
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        """
        Fit Prophet model.
        
        Args:
            y: Target series with datetime index
            X: Additional regressors (optional)
        """
        # Prepare data in Prophet format
        df = pd.DataFrame({
            'ds': y.index,
            'y': y.values
        })
        
        # Add regressors if provided
        if X is not None:
            for col in X.columns:
                df[col] = X[col].values
                self.model.add_regressor(col)
        
        self.model.fit(df)
        self.is_fitted = True
        return self
    
    def predict(self, steps: int, freq: str = 'D', X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make Prophet predictions.
        
        Args:
            steps: Number of steps to forecast
            freq: Frequency of time series
            X: Future regressors (optional)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        
        if X is not None:
            for col in X.columns:
                future[col] = X[col].values
        
        forecast = self.model.predict(future)
        return forecast['yhat'].values[-steps:]
    
    def get_params(self) -> Dict:
        """Get Prophet parameters."""
        return self.prophet_params


class ETSWrapper(BaseTimeSeriesModel):
    """Wrapper for Exponential Smoothing (ETS) model."""
    
    def __init__(self, seasonal_periods: int = None, **kwargs):
        """
        Initialize ETS model.
        
        Args:
            seasonal_periods: Number of periods in season
            **kwargs: Additional ExponentialSmoothing parameters
        """
        self.seasonal_periods = seasonal_periods
        self.ets_params = kwargs
        super().__init__(None, 'statistical')
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """Fit ETS model."""
        self.model = ExponentialSmoothing(
            y, 
            seasonal_periods=self.seasonal_periods,
            **self.ets_params
        )
        self.fitted_model = self.model.fit()
        self.is_fitted = True
        return self
    
    def predict(self, steps: int, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Make ETS predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)
    
    def get_params(self) -> Dict:
        """Get ETS parameters."""
        return {'seasonal_periods': self.seasonal_periods, **self.ets_params}


class ThetaWrapper(BaseTimeSeriesModel):
    """Wrapper for Theta model."""
    
    def __init__(self, **kwargs):
        """
        Initialize Theta model.
        
        Args:
            **kwargs: Theta model parameters
        """
        self.theta_params = kwargs
        super().__init__(None, 'statistical')
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None):
        """Fit Theta model."""
        self.model = ThetaModel(y, **self.theta_params)
        self.fitted_model = self.model.fit()
        self.is_fitted = True
        return self
    
    def predict(self, steps: int, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Make Theta predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)
    
    def get_params(self) -> Dict:
        """Get Theta parameters."""
        return self.theta_params

