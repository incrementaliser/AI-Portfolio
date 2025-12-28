"""
Main entry point for the Time Series Forecasting Pipeline

Define global data configuration and experiments as a list of dictionaries.
Each experiment specifies which model(s) to run and optional hyperparameter overrides.
"""
import sys
import os
from pathlib import Path
from typing import Any, List, Dict, Optional
from src.utils import load_config
from src.pipeline import TimeSeriesPipeline

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# ====================================================================
# GLOBAL DATA CONFIGURATION
# ====================================================================
# Data settings shared across all experiments
# Paths are relative to the script directory
DATA_CONFIG = {
    "csv_path": str(SCRIPT_DIR / "data/raw/sample_timeseries.csv"),
    "timestamp_col": "date",
    "target_cols": ["value"],  # Can be a list for multivariate
    "horizons": [1, 7, 30],  # Forecast horizons to evaluate
}


# ====================================================================
# EXPERIMENT CONFIGURATION
# ====================================================================
# Each experiment specifies:
# - model: Name of model to run (e.g., "xgboost", "arima", "lstm")
# - hyperparameters: Optional dict to override config.yaml hyperparameters
# - use_wandb: Whether to log to wandb (default: False)
# - quiet: Reduce output verbosity (default: False)

EXPERIMENTS: List[Dict] = [
    # Example: XGBoost with default hyperparameters from config.yaml
    {
        "model": "xgboost",
    },
    
    # Example: XGBoost with custom hyperparameters
    # {
    #     "model": "xgboost",
    #     "hyperparameters": {
    #         "n_estimators": 200,
    #         "max_depth": 10,
    #         "learning_rate": 0.05
    #     }
    # },
    
    # Example: Multiple models in one experiment
    # {
    #     "model": ["xgboost", "random_forest", "arima"]
    # },
    
    # Add more experiments here...
]


def validate_experiment(experiment: Dict, config: Dict) -> bool:
    """
    Validate an experiment configuration.
    
    Args:
        experiment: Experiment configuration dictionary
        config: Base configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    # Check if data file exists
    data_path = DATA_CONFIG.get('csv_path') or config.get('data', {}).get('csv_path')
    
    if data_path and not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        print("\nPlease ensure:")
        print("  1. The CSV file exists at the specified path")
        print("  2. Or update DATA_CONFIG['csv_path'] in main.py")
        return False
    
    # Validate model name(s)
    model = experiment.get('model')
    if model is None:
        print("Error: Experiment must specify 'model' key")
        return False
    
    # Get all available models from config
    available_models = []
    if 'statistical' in config.get('models', {}):
        available_models.extend(['arima', 'prophet', 'ets', 'theta'])
    if 'ml' in config.get('models', {}):
        available_models.extend(['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'ridge'])
    if 'deep_learning' in config.get('models', {}):
        available_models.extend(['lstm', 'gru', 'nbeats', 'transformer', 'tcn'])
    
    # Validate model name(s)
    if isinstance(model, str):
        models_to_check = [model]
    elif isinstance(model, list):
        models_to_check = model
    else:
        print(f"Error: 'model' must be a string or list of strings")
        return False
    
    for model_name in models_to_check:
        if model_name not in available_models:
            print(f"Error: Unknown model '{model_name}'. Available models: {available_models}")
            return False
    
    # Validate horizons
    horizons = DATA_CONFIG.get('horizons')
    if horizons is not None:
        if not isinstance(horizons, list) or not all(isinstance(h, int) and h > 0 for h in horizons):
            print(f"Error: 'horizons' in DATA_CONFIG must be a list of positive integers")
            return False
    
    return True


def run_experiment(experiment: Dict, experiment_num: int = 1, total: int = 1) -> bool:
    """
    Run a single experiment.
    
    Args:
        experiment: Experiment configuration dictionary
        experiment_num: Current experiment number (for display)
        total: Total number of experiments
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"Experiment {experiment_num}/{total}")
    print("=" * 80)
    
    # Load base configuration
    config_path = experiment.get('config', 'configs/config.yaml')
    # Resolve path relative to script directory
    if not os.path.isabs(config_path):
        config_path = str(SCRIPT_DIR / config_path)
    
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        print(f"  Resolved path: {os.path.abspath(config_path)}")
        print("Please create a config file or specify a valid path")
        return False
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False
    
    # Override config with global data configuration
    config['data']['csv_path'] = DATA_CONFIG['csv_path']
    config['data']['timestamp_col'] = DATA_CONFIG['timestamp_col']
    config['data']['target_cols'] = DATA_CONFIG['target_cols']
    config['data']['forecast_horizons'] = DATA_CONFIG['horizons']
    
    # Apply hyperparameter overrides if specified
    model_name = experiment.get('model')
    if isinstance(model_name, str):
        model_names_list = [model_name]
    else:
        model_names_list = model_name
    
    hyperparameters = experiment.get('hyperparameters', {})
    
    # Update config with hyperparameter overrides for each model
    for model in model_names_list:
        # Determine which section this model belongs to
        if model in ['arima', 'prophet', 'ets', 'theta']:
            section = 'statistical'
        elif model in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'ridge']:
            section = 'ml'
        elif model in ['lstm', 'gru', 'nbeats', 'transformer', 'tcn']:
            section = 'deep_learning'
        else:
            continue
        
        # Apply hyperparameter overrides (hyperparameters dict is flat, keyed by model name)
        if hyperparameters and model in hyperparameters:
            if section not in config['models']:
                config['models'][section] = {}
            if model not in config['models'][section]:
                config['models'][section][model] = {}
            config['models'][section][model].update(hyperparameters[model])
    
    # Validate experiment
    if not validate_experiment(experiment, config):
        return False
    
    # Print configuration if not quiet
    if not experiment.get('quiet', False):
        print("\nConfiguration:")
        print("-" * 80)
        print(f"  Data: {DATA_CONFIG['csv_path']}")
        print(f"  Timestamp: {DATA_CONFIG['timestamp_col']}")
        print(f"  Target(s): {DATA_CONFIG['target_cols']}")
        print(f"  Horizons: {DATA_CONFIG['horizons']}")
        print(f"  Model(s): {experiment.get('model')}")
        if hyperparameters:
            print(f"  Hyperparameter overrides: {hyperparameters}")
        print(f"  Wandb: {experiment.get('use_wandb', False)}")
        print("-" * 80)
    
    # Determine which models to enable in config (disable others)
    models_to_run = model_names_list
    
    # Temporarily disable models not in this experiment
    original_config = {}
    for section in ['statistical', 'ml', 'deep_learning']:
        if section in config.get('models', {}):
            original_config[section] = config['models'][section].copy()
            # Disable all models in this section
            for model_key in list(config['models'][section].keys()):
                if model_key not in models_to_run:
                    del config['models'][section][model_key]
    
    # Run pipeline
    try:
        pipeline = TimeSeriesPipeline(config)
        results = pipeline.run(
            csv_path=DATA_CONFIG['csv_path'],
            timestamp_col=DATA_CONFIG['timestamp_col'],
            target_cols=DATA_CONFIG['target_cols'],
            model_names=models_to_run,
            use_wandb=experiment.get('use_wandb', False)
        )
        
        # Restore original config
        for section in original_config:
            config['models'][section] = original_config[section]
        
        if results:
            print("\n✓ Experiment completed successfully!")
            print(f"✓ Trained {len(results)} model(s)")
            return True
        else:
            print("\nExperiment completed with no successful models")
            return False
            
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        return False
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        if not experiment.get('quiet', False):
            import traceback
            traceback.print_exc()
        return False


def main():
    """
    Main execution function.
    
    Runs all experiments defined in the EXPERIMENTS list.
    """
    if not EXPERIMENTS:
        print("=" * 80)
        print("No experiments defined!")
        print("=" * 80)
        print("\nPlease define experiments in the EXPERIMENTS list at the top of main.py")
        print("\nExample:")
        print("""
EXPERIMENTS = [
    {
        "model": "xgboost",
    },
    {
        "model": "xgboost",
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 10
        }
    }
]
        """)
        sys.exit(1)
    
    print("=" * 80)
    print("Time Series Forecasting Pipeline")
    print("=" * 80)
    print(f"\nData Configuration:")
    print(f"  File: {DATA_CONFIG['csv_path']}")
    print(f"  Timestamp: {DATA_CONFIG['timestamp_col']}")
    print(f"  Target(s): {DATA_CONFIG['target_cols']}")
    print(f"  Horizons: {DATA_CONFIG['horizons']}")
    print(f"\nRunning {len(EXPERIMENTS)} experiment(s)...")
    
    # Run all experiments
    results = []
    for idx, experiment in enumerate(EXPERIMENTS):
        success = run_experiment(experiment, experiment_num=idx, total=len(EXPERIMENTS))
        results.append(success)
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENTS SUMMARY")
    print("=" * 80)
    successful = sum(results)
    failed = len(results) - successful
    
    for idx, (experiment, success) in enumerate(zip(EXPERIMENTS, results), 1):
        status = "✓" if success else "✗"
        model_name = experiment.get('model', 'Unknown')
        print(f"{status} Experiment {idx}: {model_name}")
    
    print(f"\nTotal: {len(EXPERIMENTS)} experiments")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print("=" * 80)
    
    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
