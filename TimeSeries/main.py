"""
Main entry point for the Time Series Forecasting Pipeline

Define global data configuration and experiments as a list of dictionaries.
Each experiment specifies which model(s) to run and optional hyperparameter overrides.
"""
import sys
import os
import random
import numpy as np
from pathlib import Path
from typing import Any, List, Dict, Optional
from src.utils import load_config
from src.pipeline import TimeSeriesPipeline

# Set random seeds for reproducibility
RANDOM_SEED = 47
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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
    # XGBoost experiment
    {
        "model": "xgboost",
    },
    
    # Prophet experiment for comparison
    {
        "model": "prophet",
    },
    
    # Example: XGBoost with custom hyperparameters
    # {
    #     "model": "xgboost",
    #     "hyperparameters": {
    #         "xgboost": {
    #             "n_estimators": 200,
    #             "max_depth": 10,
    #             "learning_rate": 0.05
    #         }
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
        if isinstance(model_name, list):
            model_name = ", ".join(model_name)
        print(f"{status} Experiment {idx}: {model_name}")
    
    print(f"\nTotal: {len(EXPERIMENTS)} experiments")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    # Compare results if any successful experiments
    if successful > 0:
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        compare_experiment_results(EXPERIMENTS, results)
    
    print("=" * 80)
    
    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)


def compare_experiment_results(experiments: List[Dict], results: List[bool]):
    """
    Compare results from multiple experiments and display in a comprehensive table.
    
    Args:
        experiments: List of experiment configurations
        results: List of success/failure status for each experiment
    """
    import json
    from pathlib import Path
    
    # Load metrics from saved results
    # Use paths relative to TimeSeries directory (SCRIPT_DIR)
    possible_paths = []
    
    # Get path from config (preferred method)
    try:
        config_path = experiments[0].get('config', 'configs/config.yaml')
        if not os.path.isabs(config_path):
            config_path = str(SCRIPT_DIR / config_path)
        config = load_config(config_path)
        metrics_dir = config.get('paths', {}).get('metrics_dir', 'results/metrics/')
        
        # Resolve path relative to script directory
        if not os.path.isabs(metrics_dir):
            possible_paths.append(Path(SCRIPT_DIR / metrics_dir) / "results_summary.json")
        else:
            possible_paths.append(Path(metrics_dir) / "results_summary.json")
    except Exception as e:
        pass  # Use fallback path
    
    # Fallback: use default path relative to script directory
    possible_paths.append(SCRIPT_DIR / "results" / "metrics" / "results_summary.json")
    
    metrics_file = None
    for path in possible_paths:
        if path.exists():
            metrics_file = path
            break
    
    if metrics_file is None or not metrics_file.exists():
        print(f"  No results file found for comparison")
        print(f"  Checked paths:")
        for path in possible_paths[:5]:  # Show first 5
            print(f"    - {path} (exists: {path.exists()})")
        print(f"  Current working directory: {Path.cwd()}")
        print(f"  Script directory: {SCRIPT_DIR}")
        return
    
    try:
        with open(metrics_file, 'r') as f:
            all_results = json.load(f)
        
        # Get successful model names with mapping
        model_display_map = {
            'xgboost': 'XGBoost',
            'prophet': 'Prophet',
            'arima': 'ARIMA',
            'random_forest': 'RandomForest',
            'gradient_boosting': 'GradientBoosting',
            'lightgbm': 'LightGBM',
            'ridge': 'Ridge',
            'lstm': 'LSTM',
            'gru': 'GRU',
            'nbeats': 'NBEATS',
            'transformer': 'Transformer',
            'tcn': 'TCN',
            'ets': 'ETS',
            'theta': 'Theta'
        }
        
        # Get all models that have results (not just from experiments)
        available_models = list(all_results.keys())
        
        if len(available_models) < 1:
            print("  No model results found for comparison")
            return
        
        # Even if only one model, show the table for clarity
        
        # Get horizons from config
        horizons = DATA_CONFIG.get('horizons', [1])
        
        # Collect all metrics for all models and horizons
        all_model_metrics = {}
        metric_names = ['mae', 'rmse', 'mape', 'smape', 'r2', 'mase']
        
        for model_name in available_models:
            if model_name in all_results:
                metrics = all_results[model_name].get('metrics', {})
                all_model_metrics[model_name] = {}
                
                for horizon in horizons:
                    all_model_metrics[model_name][horizon] = {}
                    for metric in metric_names:
                        key = f'test_h{horizon}_{metric}'
                        if key in metrics:
                            all_model_metrics[model_name][horizon][metric] = metrics[key]
        
        # Print comprehensive comparison table
        print("\n" + "=" * 100)
        print("FINAL MODEL COMPARISON TABLE")
        print("=" * 100)
        
        # Table for each horizon
        for horizon in horizons:
            print(f"\n{'Forecast Horizon: ' + str(horizon) + ' steps ahead':^100}")
            print("-" * 100)
            
            # Header
            header = f"{'Model':<18}"
            header += f"{'MAE':>12}  {'RMSE':>12}  {'MAPE':>10}  {'SMAPE':>10}  {'R²':>10}  {'MASE':>10}"
            print(header)
            print("-" * 100)
            
            # Data rows
            for model_name in sorted(available_models):
                if model_name in all_model_metrics and horizon in all_model_metrics[model_name]:
                    metrics = all_model_metrics[model_name][horizon]
                    row = f"{model_name:<18}"
                    
                    # Format each metric
                    mae = metrics.get('mae')
                    rmse = metrics.get('rmse')
                    mape = metrics.get('mape')
                    smape = metrics.get('smape')
                    r2 = metrics.get('r2')
                    mase = metrics.get('mase')
                    
                    row += f"{mae:>12.4f}  " if mae is not None else f"{'N/A':>12}  "
                    row += f"{rmse:>12.4f}  " if rmse is not None else f"{'N/A':>12}  "
                    row += f"{mape:>10.2f}  " if mape is not None else f"{'N/A':>10}  "
                    row += f"{smape:>10.2f}  " if smape is not None else f"{'N/A':>10}  "
                    row += f"{r2:>10.4f}  " if r2 is not None else f"{'N/A':>10}  "
                    row += f"{mase:>10.4f}" if mase is not None else f"{'N/A':>10}"
                    
                    print(row)
            
            # Summary: Best models for each metric
            print("-" * 100)
            print("Best Models (lower is better for MAE/RMSE/MAPE/SMAPE/MASE, higher is better for R²):")
            
            # Collect valid metrics for comparison
            valid_metrics = {}
            for model_name in available_models:
                if model_name in all_model_metrics and horizon in all_model_metrics[model_name]:
                    metrics = all_model_metrics[model_name][horizon]
                    valid_metrics[model_name] = metrics
            
            if valid_metrics:
                # Best MAE (lowest)
                best_mae = min(valid_metrics.items(), 
                              key=lambda x: x[1].get('mae') if x[1].get('mae') is not None else float('inf'))
                if best_mae[1].get('mae') is not None:
                    print(f"  MAE:   {best_mae[0]:<15} ({best_mae[1]['mae']:.4f})")
                
                # Best RMSE (lowest)
                best_rmse = min(valid_metrics.items(), 
                               key=lambda x: x[1].get('rmse') if x[1].get('rmse') is not None else float('inf'))
                if best_rmse[1].get('rmse') is not None:
                    print(f"  RMSE:  {best_rmse[0]:<15} ({best_rmse[1]['rmse']:.4f})")
                
                # Best MAPE (lowest)
                best_mape = min(valid_metrics.items(), 
                               key=lambda x: x[1].get('mape') if x[1].get('mape') is not None else float('inf'))
                if best_mape[1].get('mape') is not None:
                    print(f"  MAPE:  {best_mape[0]:<15} ({best_mape[1]['mape']:.2f}%)")
                
                # Best SMAPE (lowest)
                best_smape = min(valid_metrics.items(), 
                                key=lambda x: x[1].get('smape') if x[1].get('smape') is not None else float('inf'))
                if best_smape[1].get('smape') is not None:
                    print(f"  SMAPE: {best_smape[0]:<15} ({best_smape[1]['smape']:.2f}%)")
                
                # Best R² (highest)
                best_r2 = max(valid_metrics.items(), 
                             key=lambda x: x[1].get('r2') if x[1].get('r2') is not None else float('-inf'))
                if best_r2[1].get('r2') is not None:
                    print(f"  R²:    {best_r2[0]:<15} ({best_r2[1]['r2']:.4f})")
                
                # Best MASE (lowest)
                best_mase = min(valid_metrics.items(), 
                               key=lambda x: x[1].get('mase') if x[1].get('mase') is not None else float('inf'))
                if best_mase[1].get('mase') is not None:
                    print(f"  MASE:  {best_mase[0]:<15} ({best_mase[1]['mase']:.4f})")
        
        print("\n" + "=" * 100)
        print(f"Note: Full results saved to: {SCRIPT_DIR / 'results' / 'metrics' / 'results_summary.json'}")
        print(f"      Visualizations saved to: {SCRIPT_DIR / 'results' / 'figures'}")
        print("=" * 100)
        
    except Exception as e:
        print(f"  Error comparing results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
