"""
Main entry point for the NLP Classification Pipeline

Define global data configuration and experiments as a list of dictionaries.
Each experiment specifies which model(s) to run and optional hyperparameter overrides.
"""
import sys
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict
from src.utils import load_config
from src.pipeline import NLPClassificationPipeline

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
# Use forward slashes for glob patterns (works on Windows too)
DATA_CONFIG = {
    "data_path": str(SCRIPT_DIR / "data").replace('\\', '/') + '/**/',
}


# ====================================================================
# EXPERIMENT CONFIGURATION
# ====================================================================
# Each experiment specifies:
# - model: Name of model to run (e.g., "logistic_regression", "naive_bayes_multinomial")
# - hyperparameters: Optional dict to override config.yaml hyperparameters
# - use_wandb: Whether to log to wandb (default: False)
# - quiet: Reduce output verbosity (default: False)

EXPERIMENTS: List[Dict] = [
    # Logistic Regression experiment
    {
        "model": "logistic_regression",
    },
    
    # Naive Bayes Multinomial experiment
    {
        "model": "naive_bayes_multinomial",
    },
    
    # Example: Logistic Regression with custom hyperparameters
    # {
    #     "model": "logistic_regression",
    #     "hyperparameters": {
    #         "logistic_regression": {
    #             "C": 0.5,
    #             "max_iter": 2000
    #         }
    #     }
    # },
    
    # Example: Multiple models in one experiment
    # {
    #     "model": ["logistic_regression", "naive_bayes_multinomial"]
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
    # Validate model name(s)
    model = experiment.get('model')
    if model is None:
        print("Error: Experiment must specify 'model' key")
        return False
    
    # Get all available models from config
    available_models = []
    if 'ml' in config.get('models', {}):
        available_models.extend(['logistic_regression', 'naive_bayes_multinomial', 
                                'naive_bayes_bernoulli', 'naive_bayes_gaussian'])
    
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
    config['data']['data_path'] = DATA_CONFIG['data_path']
    
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
        if model in ['logistic_regression', 'naive_bayes_multinomial', 
                     'naive_bayes_bernoulli', 'naive_bayes_gaussian']:
            section = 'ml'
        else:
            continue
        
        # Apply hyperparameter overrides
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
        print(f"  Data: {DATA_CONFIG['data_path']}")
        print(f"  Model(s): {experiment.get('model')}")
        if hyperparameters:
            print(f"  Hyperparameter overrides: {hyperparameters}")
        print(f"  Wandb: {experiment.get('use_wandb', False)}")
        print("-" * 80)
    
    # Determine which models to enable in config (disable others)
    models_to_run = model_names_list
    
    # Temporarily disable models not in this experiment
    original_config = {}
    if 'ml' in config.get('models', {}):
        original_config['ml'] = config['models']['ml'].copy()
        # Disable all models in this section
        for model_key in list(config['models']['ml'].keys()):
            if model_key not in models_to_run:
                del config['models']['ml'][model_key]
    
    # Run pipeline
    try:
        pipeline = NLPClassificationPipeline(config)
        results = pipeline.run(
            data_path=DATA_CONFIG['data_path'],
            model_names=models_to_run,
            use_wandb=experiment.get('use_wandb', False)
        )
        
        # Restore original config
        if 'ml' in original_config:
            config['models']['ml'] = original_config['ml']
        
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
        sys.exit(1)
    
    print("=" * 80)
    print("NLP Classification Pipeline")
    print("=" * 80)
    print(f"\nData Configuration:")
    print(f"  Path: {DATA_CONFIG['data_path']}")
    print(f"\nRunning {len(EXPERIMENTS)} experiment(s)...")
    
    # Run all experiments
    results = []
    for idx, experiment in enumerate(EXPERIMENTS):
        success = run_experiment(experiment, experiment_num=idx + 1, total=len(EXPERIMENTS))
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
        
        # Get all models that have results
        available_models = list(all_results.keys())
        
        if len(available_models) < 1:
            print("  No model results found for comparison")
            return
        
        # Collect all metrics for all models
        all_model_metrics = {}
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for model_name in available_models:
            if model_name in all_results:
                metrics = all_results[model_name].get('metrics', {})
                all_model_metrics[model_name] = {}
                
                for metric in metric_names:
                    key = f'test_{metric}'
                    if key in metrics:
                        all_model_metrics[model_name][metric] = metrics[key]
        
        # Print comprehensive comparison table
        print("\n" + "=" * 100)
        print("FINAL MODEL COMPARISON TABLE")
        print("=" * 100)
        
        # Header
        header = f"{'Model':<25}"
        header += f"{'Accuracy':>12}  {'Precision':>12}  {'Recall':>12}  {'F1':>12}  {'ROC-AUC':>12}"
        print(header)
        print("-" * 100)
        
        # Data rows
        for model_name in sorted(available_models):
            if model_name in all_model_metrics:
                metrics = all_model_metrics[model_name]
                row = f"{model_name:<25}"
                
                # Format each metric
                accuracy = metrics.get('accuracy')
                precision = metrics.get('precision')
                recall = metrics.get('recall')
                f1 = metrics.get('f1')
                roc_auc = metrics.get('roc_auc')
                
                row += f"{accuracy:>12.4f}  " if accuracy is not None else f"{'N/A':>12}  "
                row += f"{precision:>12.4f}  " if precision is not None else f"{'N/A':>12}  "
                row += f"{recall:>12.4f}  " if recall is not None else f"{'N/A':>12}  "
                row += f"{f1:>12.4f}  " if f1 is not None else f"{'N/A':>12}  "
                row += f"{roc_auc:>12.4f}" if roc_auc is not None else f"{'N/A':>12}"
                
                print(row)
        
        # Summary: Best models for each metric
        print("-" * 100)
        print("Best Models (higher is better for all metrics):")
        
        # Collect valid metrics for comparison
        valid_metrics = {}
        for model_name in available_models:
            if model_name in all_model_metrics:
                valid_metrics[model_name] = all_model_metrics[model_name]
        
        if valid_metrics:
            # Best Accuracy (highest)
            best_accuracy = max(valid_metrics.items(), 
                               key=lambda x: x[1].get('accuracy') if x[1].get('accuracy') is not None else float('-inf'))
            if best_accuracy[1].get('accuracy') is not None:
                print(f"  Accuracy:  {best_accuracy[0]:<20} ({best_accuracy[1]['accuracy']:.4f})")
            
            # Best Precision (highest)
            best_precision = max(valid_metrics.items(), 
                                key=lambda x: x[1].get('precision') if x[1].get('precision') is not None else float('-inf'))
            if best_precision[1].get('precision') is not None:
                print(f"  Precision: {best_precision[0]:<20} ({best_precision[1]['precision']:.4f})")
            
            # Best Recall (highest)
            best_recall = max(valid_metrics.items(), 
                             key=lambda x: x[1].get('recall') if x[1].get('recall') is not None else float('-inf'))
            if best_recall[1].get('recall') is not None:
                print(f"  Recall:    {best_recall[0]:<20} ({best_recall[1]['recall']:.4f})")
            
            # Best F1 (highest)
            best_f1 = max(valid_metrics.items(), 
                         key=lambda x: x[1].get('f1') if x[1].get('f1') is not None else float('-inf'))
            if best_f1[1].get('f1') is not None:
                print(f"  F1:        {best_f1[0]:<20} ({best_f1[1]['f1']:.4f})")
            
            # Best ROC-AUC (highest)
            best_roc_auc = max(valid_metrics.items(), 
                              key=lambda x: x[1].get('roc_auc') if x[1].get('roc_auc') is not None else float('-inf'))
            if best_roc_auc[1].get('roc_auc') is not None:
                print(f"  ROC-AUC:   {best_roc_auc[0]:<20} ({best_roc_auc[1]['roc_auc']:.4f})")
        
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

