"""
Time series data loading and preprocessing module
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Tuple, Union
import os
import json


class TimeSeriesDataLoader:
    """Data loader for time series forecasting with support for univariate and multivariate series."""
    
    def __init__(self, config: Dict):
        """
        Initialize the time series data loader.
        
        Args:
            config: Configuration dictionary containing data parameters
        """
        self.config = config
        self.scaler = None
        self.data = None
        self.feature_names = []
        
    def load_csv(
        self, 
        filepath: str, 
        timestamp_col: str, 
        target_cols: Union[str, List[str]],
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Load time series data from CSV file.
        
        Args:
            filepath: Path to CSV file
            timestamp_col: Name of timestamp column
            target_cols: Name(s) of target column(s)
            parse_dates: Whether to parse dates
            
        Returns:
            DataFrame with datetime index and target columns
        """
        df = pd.read_csv(filepath)
        
        # Parse timestamp column
        if parse_dates:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Set timestamp as index
        df = df.set_index(timestamp_col)
        df = df.sort_index()
        
        # Ensure target_cols is a list
        if isinstance(target_cols, str):
            target_cols = [target_cols]
        
        # Validate target columns exist
        for col in target_cols:
            if col not in df.columns:
                raise ValueError(f"Target column '{col}' not found in CSV")
        
        self.data = df
        self.feature_names = target_cols
        
        return df
    
    def handle_missing_values(
        self, 
        data: pd.DataFrame, 
        strategy: str = 'interpolate'
    ) -> pd.DataFrame:
        """
        Handle missing values in time series data.
        
        Args:
            data: Input DataFrame
            strategy: Strategy for handling missing values ('interpolate', 'ffill', 'bfill', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        if strategy == 'interpolate':
            return data.interpolate(method='time')
        elif strategy == 'ffill':
            return data.fillna(method='ffill')
        elif strategy == 'bfill':
            return data.fillna(method='bfill')
        elif strategy == 'drop':
            return data.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def add_temporal_features(
        self, 
        data: pd.DataFrame, 
        features: List[str] = None
    ) -> pd.DataFrame:
        """
        Add temporal features to the dataset.
        
        Args:
            data: Input DataFrame with datetime index
            features: List of features to add (hour, day, day_of_week, month, quarter, year)
            
        Returns:
            DataFrame with added temporal features
        """
        if features is None:
            features = []
        
        df = data.copy()
        
        if 'hour' in features and hasattr(df.index, 'hour'):
            df['hour'] = df.index.hour
        if 'day' in features:
            df['day'] = df.index.day
        if 'day_of_week' in features:
            df['day_of_week'] = df.index.dayofweek
        if 'month' in features:
            df['month'] = df.index.month
        if 'quarter' in features:
            df['quarter'] = df.index.quarter
        if 'year' in features:
            df['year'] = df.index.year
        
        return df
    
    def add_lag_features(
        self, 
        data: pd.DataFrame, 
        target_col: str, 
        lags: List[int]
    ) -> pd.DataFrame:
        """
        Add lag features for a target column.
        
        Args:
            data: Input DataFrame
            target_col: Column to create lags for
            lags: List of lag values (e.g., [1, 7, 30])
            
        Returns:
            DataFrame with lag features added
        """
        df = data.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def add_rolling_features(
        self, 
        data: pd.DataFrame, 
        target_col: str, 
        windows: List[int],
        statistics: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
        """
        Add rolling window statistics features.
        
        Args:
            data: Input DataFrame
            target_col: Column to calculate rolling statistics for
            windows: List of window sizes
            statistics: List of statistics to calculate ('mean', 'std', 'min', 'max')
            
        Returns:
            DataFrame with rolling features added
        """
        df = data.copy()
        
        for window in windows:
            if 'mean' in statistics:
                df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            if 'std' in statistics:
                df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            if 'min' in statistics:
                df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            if 'max' in statistics:
                df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        return df
    
    def create_sequences(
        self, 
        data: np.ndarray, 
        lookback_window: int, 
        forecast_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for supervised learning from time series data.
        
        Args:
            data: Input array of shape (n_samples, n_features)
            lookback_window: Number of time steps to look back
            forecast_horizon: Number of time steps to forecast
            
        Returns:
            Tuple of (X, y) where X has shape (n_sequences, lookback_window, n_features)
            and y has shape (n_sequences, forecast_horizon, n_features)
        """
        X, y = [], []
        
        for i in range(len(data) - lookback_window - forecast_horizon + 1):
            X.append(data[i:i + lookback_window])
            y.append(data[i + lookback_window:i + lookback_window + forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def temporal_train_test_split(
        self, 
        data: pd.DataFrame, 
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/validation/test sets while preserving temporal order.
        
        Args:
            data: Input DataFrame with temporal index
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            
        Returns:
            Dictionary containing train, validation, and test DataFrames
        """
        n = len(data)
        test_idx = int(n * (1 - test_size))
        train_val_data = data.iloc[:test_idx]
        test_data = data.iloc[test_idx:]
        
        n_train_val = len(train_val_data)
        val_idx = int(n_train_val * (1 - val_size))
        train_data = train_val_data.iloc[:val_idx]
        val_data = train_val_data.iloc[val_idx:]
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def scale_data(
        self, 
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame = None,
        test_data: pd.DataFrame = None,
        method: str = 'standard',
        columns: List[str] = None
    ) -> Tuple[pd.DataFrame, ...]:
        """
        Scale time series data, fitting only on training data.
        
        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame (optional)
            test_data: Test DataFrame (optional)
            method: Scaling method ('standard', 'minmax', 'robust')
            columns: Specific columns to scale (None means all numeric columns)
            
        Returns:
            Tuple of scaled DataFrames (train, val, test)
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Determine columns to scale
        if columns is None:
            columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create copies
        train_scaled = train_data.copy()
        
        # Fit and transform training data
        train_scaled[columns] = self.scaler.fit_transform(train_data[columns])
        
        results = [train_scaled]
        
        # Transform validation and test data if provided
        if val_data is not None:
            val_scaled = val_data.copy()
            val_scaled[columns] = self.scaler.transform(val_data[columns])
            results.append(val_scaled)
        
        if test_data is not None:
            test_scaled = test_data.copy()
            test_scaled[columns] = self.scaler.transform(test_data[columns])
            results.append(test_scaled)
        
        return tuple(results)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            data: Scaled data
            
        Returns:
            Data in original scale
        """
        if self.scaler is None:
            return data
        return self.scaler.inverse_transform(data)
    
    def prepare_data(
        self,
        filepath: str = None,
        timestamp_col: str = None,
        target_cols: Union[str, List[str]] = None,
        apply_scaling: bool = True,
        add_features: List[str] = None
    ) -> Dict:
        """
        Complete data preparation pipeline for time series forecasting.
        
        Args:
            filepath: Path to CSV file (if None, uses config)
            timestamp_col: Name of timestamp column (if None, uses config)
            target_cols: Target column(s) (if None, uses config)
            apply_scaling: Whether to apply scaling
            add_features: List of features to add
            
        Returns:
            Dictionary containing processed train/val/test data
        """
        # Use config values if not provided
        if filepath is None:
            filepath = self.config['data'].get('csv_path')
        if timestamp_col is None:
            timestamp_col = self.config['data'].get('timestamp_col', 'date')
        if target_cols is None:
            target_cols = self.config['data'].get('target_cols', ['value'])
        
        # Load data
        print(f"Loading data from {filepath}...")
        df = self.load_csv(filepath, timestamp_col, target_cols)
        
        # Handle missing values
        missing_strategy = self.config.get('preprocessing', {}).get('missing_values', 'interpolate')
        df = self.handle_missing_values(df, missing_strategy)
        
        # Add features if specified
        if add_features is not None:
            temporal_features = [f for f in add_features if f in ['hour', 'day', 'day_of_week', 'month', 'quarter', 'year']]
            if temporal_features:
                df = self.add_temporal_features(df, temporal_features)
        
        # Split data temporally
        test_size = self.config['data'].get('test_size', 0.2)
        val_size = self.config['data'].get('validation_size', 0.1)
        splits = self.temporal_train_test_split(df, test_size, val_size)
        
        # Apply scaling if requested
        if apply_scaling:
            scaling_method = self.config.get('preprocessing', {}).get('scaling', 'standard')
            train_scaled, val_scaled, test_scaled = self.scale_data(
                splits['train'], 
                splits['val'], 
                splits['test'],
                method=scaling_method,
                columns=target_cols
            )
            
            return {
                'train': train_scaled,
                'val': val_scaled,
                'test': test_scaled,
                'train_raw': splits['train'],
                'val_raw': splits['val'],
                'test_raw': splits['test']
            }
        
        return splits
    
    def save_processed_data(self, data_dict: Dict, path: str):
        """
        Save processed data to disk as CSV files with JSON metadata.
        
        Args:
            data_dict: Dictionary of processed data (DataFrames)
            path: Directory path to save data
        """
        os.makedirs(path, exist_ok=True)
        
        # Save metadata about what splits exist
        metadata = {
            'splits': list(data_dict.keys()),
            'timestamp_col': 'date',
            'saved_at': pd.Timestamp.now().isoformat()
        }
        
        # Save each DataFrame as CSV
        for key, df in data_dict.items():
            if isinstance(df, pd.DataFrame):
                csv_path = os.path.join(path, f'{key}.csv')
                df.to_csv(csv_path)
                print(f"  Saved {key} to {csv_path}")
        
        # Save metadata as JSON
        metadata_path = os.path.join(path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved metadata to {metadata_path}")
    
    def load_processed_data(self, path: str) -> Dict:
        """
        Load processed data from disk (CSV files with JSON metadata).
        
        Args:
            path: Directory path containing processed data
            
        Returns:
            Dictionary of processed data (DataFrames)
        """
        # Load metadata
        metadata_path = os.path.join(path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load each CSV file
        data_dict = {}
        for split_name in metadata['splits']:
            csv_path = os.path.join(path, f'{split_name}.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                data_dict[split_name] = df
        
        return data_dict
