"""
Main entry point for the NLP Classification Pipeline

Define global data configuration and experiments as a list of dictionaries.
Each experiment specifies which model(s) to run and optional hyperparameter overrides.

Supports:
- Traditional ML models (Logistic Regression, Naive Bayes)
- Transformer-based LM models (BERT, DistilBERT, RoBERTa)
- Accelerate-based training (Multi-GPU, mixed precision)
- Unsloth LoRA fine-tuning (Efficient LLM fine-tuning)
- Prompt-based LLM models (Zero-shot, Few-shot, Chain-of-Thought)
- DSpy prompt optimization (Learnable prompts)
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
# - model: Name of model to run (see available models below)
# - hyperparameters: Optional dict to override config.yaml hyperparameters
# - use_mlflow: Whether to log to MLflow (default: False)
# - quiet: Reduce output verbosity (default: False)
#
# Available ML models:
#   - "logistic_regression"
#   - "naive_bayes_multinomial"
#   - "naive_bayes_bernoulli"
#   - "naive_bayes_gaussian"
#
# Available LM models (require transformers library):
#   - "bert_base"       - BERT base uncased (~110M params)
#   - "distilbert"      - DistilBERT base uncased (~66M params, faster)
#   - "roberta_base"    - RoBERTa base (~125M params)
#
# Available Prompting models (require langchain + huggingface_hub):
#   HuggingFace API (no local GPU needed):
#   - "zero_shot_mistral"  - Zero-shot with Mistral 7B
#   - "few_shot_mistral"   - Few-shot with Mistral 7B
#   - "cot_mistral"        - Chain-of-thought with Mistral 7B
#   Local inference (requires GPU):
#   - "zero_shot_phi3"     - Zero-shot with Phi-3 Mini
#   - "few_shot_phi3"      - Few-shot with Phi-3 Mini
#   - "cot_phi3"           - Chain-of-thought with Phi-3 Mini
#
# Available Deep Learning models (require torch):
#   - "mlp"           - Simple MLP with averaged embeddings (fastest)
#   - "deep_mlp"      - Deeper MLP with batch norm and residual connections
#   - "cnn"           - TextCNN (Kim 2014) with multi-filter sizes
#   - "deep_cnn"      - Deeper CNN with stacked layers
#   - "lstm"          - LSTM (requires pytorch-lightning)
#   - "bilstm"        - Bidirectional LSTM
#   - "gru"           - GRU (faster than LSTM)
#   - "bigru"         - Bidirectional GRU
#   - "bilstm_attention"  - BiLSTM with additive attention
#   - "transformer_tiny"  - Custom Transformer (128d, 2 layers)
#   - "transformer_small" - Custom Transformer (256d, 4 layers)
#   - "transformer_base"  - Custom Transformer (512d, 6 layers)
#
# Available Accelerate models (require accelerate):
#   - "bert_accelerate"      - BERT with Accelerate (multi-GPU, FP16)
#   - "distilbert_accelerate" - DistilBERT with Accelerate
#   - "roberta_accelerate"   - RoBERTa with Accelerate
#
# Available Unsloth LoRA models (require unsloth + trl):
#   - "llama_lora"    - Llama 3.2 1B with LoRA (4-bit, efficient)
#   - "phi_lora"      - Phi-3.5 Mini with LoRA
#   - "mistral_lora"  - Mistral 7B with LoRA
#
# Available DSpy models (require dspy-ai):
#   - "dspy_predict"  - DSpy basic prediction with prompt optimization
#   - "dspy_cot"      - DSpy chain-of-thought with optimization
#   - "dspy_mipro"    - DSpy with MIPRO optimizer (slower, better)

EXPERIMENTS: List[Dict] = [
    # Logistic Regression experiment
    # {
    #     "model": "logistic_regression",
    # },
    
    # Naive Bayes Multinomial experiment
    # {
    #     "model": "naive_bayes_multinomial",
    # },
    
    # =========================================================
    # LANGUAGE MODEL EXPERIMENTS (uncomment to run)
    # =========================================================
    # Note: LM experiments require:
    #   1. transformers and torch installed (pip install transformers torch)
    #   2. GPU recommended for reasonable training times
    #   3. More training time than ML models
    
    # DistilBERT experiment (recommended for first LM experiment - faster)
    # {
    #     "model": "distilbert",
    # },
    
    # BERT experiment
    {
        "model": "bert_base",
    },
    
    # RoBERTa experiment
    # {
    #     "model": "roberta_base",
    # },
    
    # DistilBERT with custom hyperparameters
    # {
    #     "model": "distilbert",
    #     "hyperparameters": {
    #         "distilbert": {
    #             "epochs": 2,
    #             "batch_size": 16,
    #             "learning_rate": 2e-5,
    #         }
    #     }
    # },
    
    # =========================================================
    # PROMPTING EXPERIMENTS (uncomment to run)
    # =========================================================
    # Note: Prompting experiments require:
    #   1. langchain and huggingface_hub installed
    #   2. HuggingFace API token (set HF_TOKEN environment variable)
    #   3. For local models: transformers, torch, and GPU
    
    # Zero-shot with Mistral 7B (HuggingFace API - no local GPU needed)
    # {
    #     "model": "zero_shot_mistral",
    # },
    
    # Few-shot with Mistral 7B (selects examples from training data)
    # {
    #     "model": "few_shot_mistral",
    # },
    
    # Chain-of-thought with Mistral 7B (step-by-step reasoning)
    # {
    #     "model": "cot_mistral",
    # },
    
    # Local models (requires GPU)
    # {
    #     "model": "zero_shot_phi3",
    # },
    
    # Compare prompting techniques
    # {
    #     "model": ["zero_shot_mistral", "few_shot_mistral", "cot_mistral"],
    # },
    
    # Compare ML vs Prompting
    # {
    #     "model": ["logistic_regression", "zero_shot_mistral"],
    # },
    
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
    
    # =========================================================
    # ACCELERATE EXPERIMENTS (uncomment to run)
    # =========================================================
    # Multi-GPU, mixed precision training with HuggingFace Accelerate
    # Requires: pip install accelerate
    
    # BERT with Accelerate (FP16, gradient accumulation)
    # {
    #     "model": "bert_accelerate",
    # },
    
    # DistilBERT with Accelerate (faster)
    # {
    #     "model": "distilbert_accelerate",
    # },
    
    # =========================================================
    # UNSLOTH LORA EXPERIMENTS (uncomment to run)
    # =========================================================
    # Efficient LoRA fine-tuning, 2-5x faster with 60% less memory
    # Requires: pip install unsloth trl
    
    # Llama 3.2 1B with LoRA (small, fast, good for testing)
    # {
    #     "model": "llama_lora",
    # },
    
    # Phi-3.5 Mini with LoRA (medium size, good quality)
    # {
    #     "model": "phi_lora",
    # },
    
    # Mistral 7B with LoRA (large, best quality, needs ~16GB VRAM)
    # {
    #     "model": "mistral_lora",
    # },
    
    # =========================================================
    # DSPY EXPERIMENTS (uncomment to run)
    # =========================================================
    # Learnable prompt optimization using DSpy
    # Requires: pip install dspy-ai
    
    # DSpy with prompt optimization
    # {
    #     "model": "dspy_predict",
    # },
    
    # DSpy chain-of-thought with optimization
    # {
    #     "model": "dspy_cot",
    # },
    
    # =========================================================
    # DEEP LEARNING EXPERIMENTS (uncomment to run)
    # =========================================================
    # Neural network models from simple to complex
    # Requires: pip install torch (pytorch-lightning for RNN/Attention)
    
    # MLP - Simple baseline (fastest deep model)
    # {
    #     "model": "mlp",
    # },
    
    # CNN - Captures n-gram patterns (Kim 2014 style)
    # {
    #     "model": "cnn",
    # },
    
    # LSTM - Sequential modeling
    # {
    #     "model": "lstm",
    # },
    
    # BiLSTM - Bidirectional context
    # {
    #     "model": "bilstm",
    # },
    
    # BiLSTM with Attention - Focus on important tokens
    # {
    #     "model": "bilstm_attention",
    # },
    
    # Custom Transformer - Small version (balanced speed/quality)
    # {
    #     "model": "transformer_small",
    # },
    
    # Compare deep learning architectures
    # {
    #     "model": ["mlp", "cnn", "bilstm", "transformer_small"],
    # },
    
    # =========================================================
    # COMPARISON EXPERIMENTS
    # =========================================================
    
    # Compare all fine-tuning approaches
    # {
    #     "model": ["distilbert", "distilbert_accelerate"],
    # },
    
    # Compare prompting approaches
    # {
    #     "model": ["zero_shot_mistral", "dspy_predict"],
    # },
    
    # Add more experiments here...
]

# ====================================================================
# MODEL CATEGORIES
# ====================================================================
# Used for validation and configuration routing
ML_MODELS = ['logistic_regression', 'naive_bayes_multinomial', 
             'naive_bayes_bernoulli', 'naive_bayes_gaussian']
DEEP_MODELS = [
    'mlp', 'deep_mlp', 'cnn', 'deep_cnn',
    'lstm', 'bilstm', 'gru', 'bigru',
    'bilstm_attention', 'bilstm_self_attention', 'bilstm_multihead_attention',
    'transformer_tiny', 'transformer_small', 'transformer_base', 'transformer'
]
LM_MODELS = ['bert_base', 'distilbert', 'roberta_base']
ACCELERATE_MODELS = ['bert_accelerate', 'distilbert_accelerate', 'roberta_accelerate']
UNSLOTH_MODELS = ['llama_lora', 'phi_lora', 'mistral_lora', 'unsloth_lora']
PROMPTING_MODELS = [
    'zero_shot_mistral', 'few_shot_mistral', 'cot_mistral',
    'zero_shot_phi3', 'few_shot_phi3', 'cot_phi3',
    'zero_shot', 'few_shot', 'chain_of_thought'
]
DSPY_MODELS = ['dspy_predict', 'dspy_cot', 'dspy_mipro', 'dspy_react']
ALL_MODELS = ML_MODELS + DEEP_MODELS + LM_MODELS + ACCELERATE_MODELS + UNSLOTH_MODELS + PROMPTING_MODELS + DSPY_MODELS


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
    
    # Build list of available models from config
    available_models = []
    
    # ML models
    if 'ml' in config.get('models', {}):
        for model_key in ML_MODELS:
            if model_key in config['models']['ml']:
                available_models.append(model_key)
    
    # LM models
    if 'lm' in config.get('models', {}):
        for model_key in LM_MODELS:
            if model_key in config['models']['lm']:
                available_models.append(model_key)
    
    # Accelerate models
    if 'accelerate' in config.get('models', {}):
        for model_key in ACCELERATE_MODELS:
            if model_key in config['models']['accelerate']:
                available_models.append(model_key)
    
    # Unsloth models
    if 'unsloth' in config.get('models', {}):
        for model_key in UNSLOTH_MODELS:
            if model_key in config['models']['unsloth']:
                available_models.append(model_key)
    
    # Prompting models
    if 'prompting' in config.get('models', {}):
        for model_key in PROMPTING_MODELS:
            if model_key in config['models']['prompting']:
                available_models.append(model_key)
    
    # DSpy models
    if 'dspy' in config.get('models', {}):
        for model_key in DSPY_MODELS:
            if model_key in config['models']['dspy']:
                available_models.append(model_key)
    
    # Deep learning models
    if 'deep' in config.get('models', {}):
        for model_key in DEEP_MODELS:
            if model_key in config['models']['deep']:
                available_models.append(model_key)
    
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
            print(f"Error: Unknown model '{model_name}'.")
            print(f"  Available models: {available_models}")
            print(f"  All supported models: {ALL_MODELS}")
            if model_name in LM_MODELS and 'lm' not in config.get('models', {}):
                print(f"  Note: LM models require lm_config.yaml to be loaded.")
            if model_name in ACCELERATE_MODELS and 'accelerate' not in config.get('models', {}):
                print(f"  Note: Accelerate models require advanced_config.yaml to be loaded.")
            if model_name in UNSLOTH_MODELS and 'unsloth' not in config.get('models', {}):
                print(f"  Note: Unsloth models require advanced_config.yaml to be loaded.")
            if model_name in PROMPTING_MODELS and 'prompting' not in config.get('models', {}):
                print(f"  Note: Prompting models require prompting_config.yaml to be loaded.")
            if model_name in DSPY_MODELS and 'dspy' not in config.get('models', {}):
                print(f"  Note: DSpy models require advanced_config.yaml to be loaded.")
            if model_name in DEEP_MODELS and 'deep' not in config.get('models', {}):
                print(f"  Note: Deep learning models require deep_config.yaml to be loaded.")
            return False
    
    return True


def load_configs_for_experiment(experiment: Dict) -> Dict:
    """
    Load and merge configuration files for an experiment.
    
    Loads the base config.yaml and optionally merges lm_config.yaml
    and prompting_config.yaml based on the models being used.
    
    Args:
        experiment: Experiment configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    # Load base configuration
    config_path = experiment.get('config', 'configs/config.yaml')
    if not os.path.isabs(config_path):
        config_path = str(SCRIPT_DIR / config_path)
    
    config = load_config(config_path)
    
    # Check which model types are requested
    model = experiment.get('model')
    if isinstance(model, str):
        model_list = [model]
    else:
        model_list = model if model else []
    
    needs_lm_config = any(m in LM_MODELS for m in model_list)
    needs_prompting_config = any(m in PROMPTING_MODELS for m in model_list)
    needs_advanced_config = any(
        m in ACCELERATE_MODELS or m in UNSLOTH_MODELS or m in DSPY_MODELS 
        for m in model_list
    )
    needs_deep_config = any(m in DEEP_MODELS for m in model_list)
    
    # Load and merge LM config if needed
    if needs_lm_config:
        lm_config_path = str(SCRIPT_DIR / 'configs' / 'lm_config.yaml')
        if os.path.exists(lm_config_path):
            lm_config = load_config(lm_config_path)
            
            # Merge LM models into main config
            if 'models' not in config:
                config['models'] = {}
            if 'lm' in lm_config.get('models', {}):
                config['models']['lm'] = lm_config['models']['lm']
            
            # Merge training config if present
            if 'training' in lm_config:
                config['training'] = lm_config['training']
        else:
            print(f"Warning: LM config not found at {lm_config_path}")
            print("  LM models may not work correctly without this config.")
    
    # Load and merge prompting config if needed
    if needs_prompting_config:
        prompting_config_path = str(SCRIPT_DIR / 'configs' / 'prompting_config.yaml')
        if os.path.exists(prompting_config_path):
            prompting_config = load_config(prompting_config_path)
            
            # Merge prompting models into main config
            if 'models' not in config:
                config['models'] = {}
            if 'prompting' in prompting_config.get('models', {}):
                config['models']['prompting'] = prompting_config['models']['prompting']
            
            # Merge inference config if present
            if 'inference' in prompting_config:
                config['inference'] = prompting_config['inference']
        else:
            print(f"Warning: Prompting config not found at {prompting_config_path}")
            print("  Prompting models may not work correctly without this config.")
    
    # Load and merge advanced config if needed (Accelerate, Unsloth, DSpy)
    if needs_advanced_config:
        advanced_config_path = str(SCRIPT_DIR / 'configs' / 'advanced_config.yaml')
        if os.path.exists(advanced_config_path):
            advanced_config = load_config(advanced_config_path)
            
            # Merge advanced models into main config
            if 'models' not in config:
                config['models'] = {}
            
            # Accelerate models
            if 'accelerate' in advanced_config.get('models', {}):
                config['models']['accelerate'] = advanced_config['models']['accelerate']
            
            # Unsloth models
            if 'unsloth' in advanced_config.get('models', {}):
                config['models']['unsloth'] = advanced_config['models']['unsloth']
            
            # DSpy models
            if 'dspy' in advanced_config.get('models', {}):
                config['models']['dspy'] = advanced_config['models']['dspy']
            
            # Merge training config if present
            if 'training' in advanced_config:
                if 'training' not in config:
                    config['training'] = {}
                config['training'].update(advanced_config['training'])
        else:
            print(f"Warning: Advanced config not found at {advanced_config_path}")
            print("  Accelerate/Unsloth/DSpy models may not work correctly without this config.")
    
    # Load and merge deep config if needed
    if needs_deep_config:
        deep_config_path = str(SCRIPT_DIR / 'configs' / 'deep_config.yaml')
        if os.path.exists(deep_config_path):
            deep_config = load_config(deep_config_path)
            
            # Merge deep models into main config
            if 'models' not in config:
                config['models'] = {}
            if 'deep' in deep_config.get('models', {}):
                config['models']['deep'] = deep_config['models']['deep']
        else:
            print(f"Warning: Deep config not found at {deep_config_path}")
            print("  Deep learning models may not work correctly without this config.")
    
    return config


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
    
    # Load and merge configurations
    try:
        config = load_configs_for_experiment(experiment)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
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
        if model in ML_MODELS:
            section = 'ml'
        elif model in LM_MODELS:
            section = 'lm'
        elif model in ACCELERATE_MODELS:
            section = 'accelerate'
        elif model in UNSLOTH_MODELS:
            section = 'unsloth'
        elif model in PROMPTING_MODELS:
            section = 'prompting'
        elif model in DSPY_MODELS:
            section = 'dspy'
        elif model in DEEP_MODELS:
            section = 'deep'
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
        
        # Indicate model types
        ml_in_exp = [m for m in model_names_list if m in ML_MODELS]
        lm_in_exp = [m for m in model_names_list if m in LM_MODELS]
        prompting_in_exp = [m for m in model_names_list if m in PROMPTING_MODELS]
        
        if ml_in_exp:
            print(f"  ML models: {ml_in_exp}")
        if lm_in_exp:
            print(f"  LM models: {lm_in_exp} (GPU recommended)")
        if prompting_in_exp:
            api_models = [m for m in prompting_in_exp if 'mistral' in m or 'llama' in m]
            local_models = [m for m in prompting_in_exp if 'phi' in m or 'tiny' in m]
            if api_models:
                print(f"  Prompting models (API): {api_models}")
            if local_models:
                print(f"  Prompting models (local): {local_models}")
        
        if hyperparameters:
            print(f"  Hyperparameter overrides: {hyperparameters}")
        print(f"  MLflow: {experiment.get('use_mlflow', False)}")
        print("-" * 80)
    
    # Determine which models to enable in config (disable others)
    models_to_run = model_names_list
    
    # Temporarily disable models not in this experiment
    original_config = {
        'ml': None, 'lm': None, 'accelerate': None, 
        'unsloth': None, 'prompting': None, 'dspy': None, 'deep': None
    }
    
    # Handle ML models
    if 'ml' in config.get('models', {}):
        original_config['ml'] = config['models']['ml'].copy()
        for model_key in list(config['models']['ml'].keys()):
            if model_key not in models_to_run:
                del config['models']['ml'][model_key]
    
    # Handle LM models
    if 'lm' in config.get('models', {}):
        original_config['lm'] = config['models']['lm'].copy()
        for model_key in list(config['models']['lm'].keys()):
            if model_key not in models_to_run:
                del config['models']['lm'][model_key]
    
    # Handle Accelerate models
    if 'accelerate' in config.get('models', {}):
        original_config['accelerate'] = config['models']['accelerate'].copy()
        for model_key in list(config['models']['accelerate'].keys()):
            if model_key not in models_to_run:
                del config['models']['accelerate'][model_key]
    
    # Handle Unsloth models
    if 'unsloth' in config.get('models', {}):
        original_config['unsloth'] = config['models']['unsloth'].copy()
        for model_key in list(config['models']['unsloth'].keys()):
            if model_key not in models_to_run:
                del config['models']['unsloth'][model_key]
    
    # Handle Prompting models
    if 'prompting' in config.get('models', {}):
        original_config['prompting'] = config['models']['prompting'].copy()
        for model_key in list(config['models']['prompting'].keys()):
            if model_key not in models_to_run:
                del config['models']['prompting'][model_key]
    
    # Handle DSpy models
    if 'dspy' in config.get('models', {}):
        original_config['dspy'] = config['models']['dspy'].copy()
        for model_key in list(config['models']['dspy'].keys()):
            if model_key not in models_to_run:
                del config['models']['dspy'][model_key]
    
    # Handle Deep learning models
    if 'deep' in config.get('models', {}):
        original_config['deep'] = config['models']['deep'].copy()
        for model_key in list(config['models']['deep'].keys()):
            if model_key not in models_to_run:
                del config['models']['deep'][model_key]
    
    # Run pipeline
    try:
        pipeline = NLPClassificationPipeline(config)
        results = pipeline.run(
            data_path=DATA_CONFIG['data_path'],
            model_names=models_to_run,
            use_mlflow=experiment.get('use_mlflow', False)
        )
        
        # Restore original config
        if original_config['ml'] is not None:
            config['models']['ml'] = original_config['ml']
        if original_config['lm'] is not None:
            config['models']['lm'] = original_config['lm']
        if original_config['accelerate'] is not None:
            config['models']['accelerate'] = original_config['accelerate']
        if original_config['unsloth'] is not None:
            config['models']['unsloth'] = original_config['unsloth']
        if original_config['prompting'] is not None:
            config['models']['prompting'] = original_config['prompting']
        if original_config['dspy'] is not None:
            config['models']['dspy'] = original_config['dspy']
        if original_config['deep'] is not None:
            config['models']['deep'] = original_config['deep']
        
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
        print("\nAvailable models:")
        print(f"  ML models: {ML_MODELS}")
        print(f"  LM models: {LM_MODELS}")
        print(f"  Prompting models: {PROMPTING_MODELS}")
        sys.exit(1)
    
    print("=" * 80)
    print("NLP Classification Pipeline")
    print("=" * 80)
    print(f"\nData Configuration:")
    print(f"  Path: {DATA_CONFIG['data_path']}")
    print(f"\nRunning {len(EXPERIMENTS)} experiment(s)...")
    
    # Check for LM/Prompting experiments and warn if needed
    all_models_in_experiments = []
    for exp in EXPERIMENTS:
        model = exp.get('model', [])
        if isinstance(model, str):
            all_models_in_experiments.append(model)
        else:
            all_models_in_experiments.extend(model)
    
    lm_experiments = [m for m in all_models_in_experiments if m in LM_MODELS]
    prompting_experiments = [m for m in all_models_in_experiments if m in PROMPTING_MODELS]
    
    if lm_experiments:
        print(f"\nNote: LM models detected: {lm_experiments}")
        print("  - GPU is recommended for faster training")
        print("  - Training may take several minutes per model")
    
    if prompting_experiments:
        print(f"\nNote: Prompting models detected: {prompting_experiments}")
        api_models = [m for m in prompting_experiments if 'mistral' in m or 'llama' in m]
        if api_models:
            print("  - API models require HF_TOKEN environment variable")
            print("  - Rate limits apply on free tier")
    
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
        
        # Create final combined visualizations from all models
        print("\n" + "=" * 80)
        print("GENERATING COMBINED VISUALIZATIONS")
        print("=" * 80)
        create_combined_visualizations(EXPERIMENTS)
    
    print("=" * 80)
    
    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)


def create_combined_visualizations(experiments: List[Dict]):
    """
    Create combined visualizations for all trained models.
    
    Args:
        experiments: List of experiment configurations
    """
    import json
    from pathlib import Path
    from src.evaluate import NLPEvaluator
    import matplotlib.pyplot as plt
    
    # Load config to get paths
    try:
        config = load_configs_for_experiment(experiments[0])
    except Exception as e:
        print(f"  Error loading config: {e}")
        return
    
    # Get paths
    metrics_dir = config.get('paths', {}).get('metrics_dir', 'results/metrics/')
    figures_dir = config.get('paths', {}).get('figures_dir', 'results/figures/')
    
    # Resolve paths relative to script directory
    if not os.path.isabs(metrics_dir):
        metrics_dir = SCRIPT_DIR / metrics_dir
    if not os.path.isabs(figures_dir):
        figures_dir = SCRIPT_DIR / figures_dir
    
    metrics_file = Path(metrics_dir) / "results_summary.json"
    
    if not metrics_file.exists():
        print(f"  No results file found at {metrics_file}")
        return
    
    try:
        # Load all saved results
        with open(metrics_file, 'r') as f:
            all_results = json.load(f)
        
        if not all_results:
            print("  No model results found")
            return
        
        print(f"  Found {len(all_results)} model(s) in results")
        
        # Reconstruct results_dict format expected by evaluator
        results_dict = {}
        for model_name, model_data in all_results.items():
            if 'predictions' in model_data:
                # Convert lists back to numpy arrays
                predictions_dict = {}
                for split_name, pred_data in model_data['predictions'].items():
                    predictions_dict[split_name] = {
                        'predictions': np.array(pred_data['predictions']),
                        'actuals': np.array(pred_data['actuals']),
                        'probabilities': np.array(pred_data['probabilities']) if pred_data.get('probabilities') is not None else None
                    }
                
                results_dict[model_name] = {
                    'metrics': model_data.get('metrics', {}),
                    'predictions': predictions_dict,
                    'model_type': model_data.get('model_type', 'unknown')
                }
        
        if not results_dict:
            print("  No valid prediction data found for visualization")
            return
        
        # Create evaluator and generate combined plots
        evaluator = NLPEvaluator(config)
        
        # Combined confusion matrices
        print("  Creating combined confusion matrices...")
        cm_fig = evaluator.plot_all_confusion_matrices(
            results_dict,
            save_path=str(figures_dir)
        )
        cm_fig.clf()
        plt.close(cm_fig)
        print(f"  ✓ Saved: {figures_dir / 'all_confusion_matrices.png'}")
        
        # Combined ROC curves
        print("  Creating combined ROC curves...")
        roc_fig = evaluator.plot_all_roc_curves(
            results_dict,
            save_path=str(figures_dir)
        )
        roc_fig.clf()
        plt.close(roc_fig)
        print(f"  ✓ Saved: {figures_dir / 'all_roc_curves.png'}")
        
        # Model comparison bar chart
        print("  Creating model comparison chart...")
        comp_fig = evaluator.compare_models(
            results_dict,
            save_path=str(figures_dir)
        )
        comp_fig.clf()
        plt.close(comp_fig)
        print(f"  ✓ Saved: {figures_dir / 'model_comparison.png'}")
        
        print("\n  All combined visualizations generated successfully!")
        
    except Exception as e:
        print(f"  Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


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
        config = load_configs_for_experiment(experiments[0])
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


