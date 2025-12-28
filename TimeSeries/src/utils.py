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


def save_results(results, path):
    """Save results to JSON file"""
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
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"\nResults saved to: {results_path}")


def print_config(config):
    """Print configuration in a readable format"""
    print("\nConfiguration:")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False))
    print("=" * 60)