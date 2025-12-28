"""
Utility functions
"""
import os
import json
import yaml


def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config):
    """Create necessary directories if they don't exist"""
    directories = [
        config['paths']['data_raw'],
        config['paths']['data_processed'],
        config['paths']['models_dir'],
        config['paths']['figures_dir'],
        config['paths']['metrics_dir']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_results(results, path, append=False):
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary of model results
        path: Directory path to save results (can be relative or absolute)
        append: If True, append to existing results instead of overwriting
    """
    # Ensure path is absolute
    if not os.path.isabs(path):
        # If relative, resolve relative to current working directory
        path = os.path.abspath(path)
    
    os.makedirs(path, exist_ok=True)
    
    # Prepare results for JSON serialization
    json_results = {}
    for model_name, result_dict in results.items():
        json_results[model_name] = {
            'metrics': {}
        }
        
        # Convert metrics to JSON-serializable format
        for metric_name, metric_value in result_dict['metrics'].items():
            if isinstance(metric_value, dict):
                json_results[model_name]['metrics'][metric_name] = metric_value
            else:
                json_results[model_name]['metrics'][metric_name] = float(metric_value)
    
    # Save to file
    results_path = os.path.join(path, 'results_summary.json')
    results_path = os.path.abspath(results_path)  # Ensure absolute path
    
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


def print_config(config):
    """Print configuration in a readable format"""
    print("\nConfiguration:")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 60)