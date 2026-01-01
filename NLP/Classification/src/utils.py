"""
Utility functions for NLP classification pipeline
"""
import os
import json
import yaml


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: dict) -> None:
    """
    Create necessary directories if they don't exist.
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config['paths']['data_raw'],
        config['paths']['data_processed'],
        config['paths']['models_dir'],
        config['paths']['figures_dir'],
        config['paths']['metrics_dir']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_results(results: dict, path: str, append: bool = False) -> None:
    """
    Save results to JSON file, including predictions for visualization.
    
    Args:
        results: Dictionary of model results
        path: Directory path to save results (can be relative or absolute)
        append: If True, append to existing results instead of overwriting
    """
    import numpy as np
    
    # Ensure path is absolute
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    
    os.makedirs(path, exist_ok=True)
    
    # Prepare results for JSON serialization
    json_results = {}
    for model_name, result_dict in results.items():
        json_results[model_name] = {
            'metrics': {},
            'model_type': result_dict.get('model_type', 'unknown')
        }
        
        # Convert metrics to JSON-serializable format
        for metric_name, metric_value in result_dict['metrics'].items():
            if isinstance(metric_value, dict):
                json_results[model_name]['metrics'][metric_name] = metric_value
            elif isinstance(metric_value, (int, float)):
                json_results[model_name]['metrics'][metric_name] = float(metric_value)
            else:
                json_results[model_name]['metrics'][metric_name] = str(metric_value)
        
        # Save predictions for visualization (convert numpy arrays to lists)
        if 'predictions' in result_dict:
            json_results[model_name]['predictions'] = {}
            for split_name, pred_data in result_dict['predictions'].items():
                json_results[model_name]['predictions'][split_name] = {
                    'predictions': pred_data['predictions'].tolist() if isinstance(pred_data['predictions'], np.ndarray) else pred_data['predictions'],
                    'actuals': pred_data['actuals'].tolist() if isinstance(pred_data['actuals'], np.ndarray) else pred_data['actuals'],
                    'probabilities': pred_data['probabilities'].tolist() if pred_data['probabilities'] is not None and isinstance(pred_data['probabilities'], np.ndarray) else pred_data.get('probabilities')
                }
    
    # Save to file
    results_path = os.path.join(path, 'results_summary.json')
    results_path = os.path.abspath(results_path)
    
    # If appending, load existing results first
    if append and os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                existing_results = json.load(f)
            # Merge new results with existing (new overwrites old for same model)
            existing_results.update(json_results)
            json_results = existing_results
        except:
            pass  # If can't load, just overwrite
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"\nResults saved to: {results_path}")


def print_config(config: dict) -> None:
    """
    Print configuration in a readable format.
    
    Args:
        config: Configuration dictionary
    """
    print("\nConfiguration:")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 60)


