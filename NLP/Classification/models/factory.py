"""
Factory for creating NLP classification models.

Supports:
- Traditional ML models (Logistic Regression, Naive Bayes)
- Deep Learning models (MLP, CNN, LSTM, GRU, Attention, Transformer)
- Transformer-based LM models (BERT, DistilBERT, RoBERTa)
- Accelerate-based training (Multi-GPU, mixed precision)
- Unsloth LoRA fine-tuning (Efficient LLM fine-tuning)
- Prompt-based models (Zero-shot, Few-shot, Chain-of-Thought)
- DSpy-based prompt optimization
"""
import numpy as np
import random
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 47
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

from models.base import BaseNLPModel
from models.ml.ml_models import (
    LogisticRegressionModel,
    MultinomialNBModel,
    BernoulliNBModel,
    GaussianNBModel
)

# LM model imports - wrapped in try/except
LM_AVAILABLE = False
try:
    from models.LMs import (
        BERTClassifier,
        DistilBERTClassifier,
        RoBERTaClassifier
    )
    LM_AVAILABLE = True
except ImportError:
    pass

# Accelerate model imports
ACCELERATE_AVAILABLE = False
try:
    from models.LMs.accelerate_trainer import (
        AccelerateTrainer,
        BERTAccelerate,
        DistilBERTAccelerate,
        RoBERTaAccelerate
    )
    ACCELERATE_AVAILABLE = True
except ImportError:
    pass

# Unsloth LoRA model imports
UNSLOTH_AVAILABLE = False
try:
    from models.LMs.unsloth_lora import (
        UnslothLoRAClassifier,
        LlamaLoRA,
        PhiLoRA,
        MistralLoRA
    )
    UNSLOTH_AVAILABLE = True
except ImportError:
    pass

# Prompting model imports
PROMPTING_AVAILABLE = False
try:
    from models.LMs.prompting import (
        ZeroShotPromptClassifier,
        FewShotPromptClassifier,
        ChainOfThoughtClassifier
    )
    PROMPTING_AVAILABLE = True
except ImportError:
    pass

# DSpy model imports
DSPY_AVAILABLE = False
try:
    from models.LMs.prompting.dspy_optimizer import (
        DSPyClassifier,
        DSPyReActClassifier
    )
    DSPY_AVAILABLE = True
except ImportError:
    pass

# Deep learning model imports (PyTorch)
DEEP_AVAILABLE = False
try:
    from models.Deep import (
        MLPClassifier,
        DeepMLPClassifier,
        CNNClassifier,
        DeepCNNClassifier,
        TransformerClassifier,
        TransformerTiny,
        TransformerSmall,
        TransformerBase
    )
    DEEP_AVAILABLE = True
except ImportError:
    pass

# Deep learning models with Lightning (LSTM, GRU, Attention)
DEEP_LIGHTNING_AVAILABLE = False
try:
    from models.Deep import (
        LSTMClassifier,
        GRUClassifier,
        BiLSTMClassifier,
        BiGRUClassifier,
        BiLSTMAttentionClassifier
    )
    DEEP_LIGHTNING_AVAILABLE = True
except ImportError:
    pass


class NLPModelFactory:
    """Factory class to create NLP classification models based on configuration."""
    
    # Mapping from config keys to model classes and display names
    ML_MODEL_MAPPING = {
        'logistic_regression': ('LogisticRegression', LogisticRegressionModel),
        'naive_bayes_multinomial': ('MultinomialNB', MultinomialNBModel),
        'naive_bayes_bernoulli': ('BernoulliNB', BernoulliNBModel),
        'naive_bayes_gaussian': ('GaussianNB', GaussianNBModel),
    }
    
    # Standard LM model mapping
    LM_MODEL_MAPPING = {}
    if LM_AVAILABLE:
        LM_MODEL_MAPPING = {
            'bert_base': ('BERT', BERTClassifier),
            'distilbert': ('DistilBERT', DistilBERTClassifier),
            'roberta_base': ('RoBERTa', RoBERTaClassifier),
        }
    
    # Accelerate-based model mapping
    ACCELERATE_MODEL_MAPPING = {}
    if ACCELERATE_AVAILABLE:
        ACCELERATE_MODEL_MAPPING = {
            'bert_accelerate': ('BERT_Accelerate', BERTAccelerate),
            'distilbert_accelerate': ('DistilBERT_Accelerate', DistilBERTAccelerate),
            'roberta_accelerate': ('RoBERTa_Accelerate', RoBERTaAccelerate),
        }
    
    # Unsloth LoRA model mapping
    UNSLOTH_MODEL_MAPPING = {}
    if UNSLOTH_AVAILABLE:
        UNSLOTH_MODEL_MAPPING = {
            'llama_lora': ('Llama_LoRA', LlamaLoRA),
            'phi_lora': ('Phi_LoRA', PhiLoRA),
            'mistral_lora': ('Mistral_LoRA', MistralLoRA),
            'unsloth_lora': ('Unsloth_LoRA', UnslothLoRAClassifier),
        }
    
    # Prompting model mapping
    PROMPTING_MODEL_MAPPING = {}
    if PROMPTING_AVAILABLE:
        PROMPTING_MODEL_MAPPING = {
            'zero_shot_mistral': ('ZeroShot_Mistral', ZeroShotPromptClassifier),
            'few_shot_mistral': ('FewShot_Mistral', FewShotPromptClassifier),
            'cot_mistral': ('CoT_Mistral', ChainOfThoughtClassifier),
            'zero_shot_phi3': ('ZeroShot_Phi3', ZeroShotPromptClassifier),
            'few_shot_phi3': ('FewShot_Phi3', FewShotPromptClassifier),
            'cot_phi3': ('CoT_Phi3', ChainOfThoughtClassifier),
            'zero_shot': ('ZeroShot', ZeroShotPromptClassifier),
            'few_shot': ('FewShot', FewShotPromptClassifier),
            'chain_of_thought': ('ChainOfThought', ChainOfThoughtClassifier),
        }
    
    # DSpy model mapping
    DSPY_MODEL_MAPPING = {}
    if DSPY_AVAILABLE:
        DSPY_MODEL_MAPPING = {
            'dspy_predict': ('DSPy_Predict', DSPyClassifier),
            'dspy_cot': ('DSPy_CoT', DSPyClassifier),
            'dspy_mipro': ('DSPy_MIPRO', DSPyClassifier),
            'dspy_react': ('DSPy_ReAct', DSPyReActClassifier),
        }
    
    # Deep learning model mapping (PyTorch)
    DEEP_MODEL_MAPPING = {}
    if DEEP_AVAILABLE:
        DEEP_MODEL_MAPPING = {
            'mlp': ('MLP', MLPClassifier),
            'deep_mlp': ('DeepMLP', DeepMLPClassifier),
            'cnn': ('CNN', CNNClassifier),
            'deep_cnn': ('DeepCNN', DeepCNNClassifier),
            'transformer_tiny': ('Transformer_Tiny', TransformerTiny),
            'transformer_small': ('Transformer_Small', TransformerSmall),
            'transformer_base': ('Transformer_Base', TransformerBase),
            'transformer': ('Transformer', TransformerClassifier),
        }
    
    # Deep learning with Lightning (LSTM, GRU, Attention)
    DEEP_LIGHTNING_MODEL_MAPPING = {}
    if DEEP_LIGHTNING_AVAILABLE:
        DEEP_LIGHTNING_MODEL_MAPPING = {
            'lstm': ('LSTM', LSTMClassifier),
            'gru': ('GRU', GRUClassifier),
            'bilstm': ('BiLSTM', BiLSTMClassifier),
            'bigru': ('BiGRU', BiGRUClassifier),
            'bilstm_attention': ('BiLSTM_Attention', BiLSTMAttentionClassifier),
            'bilstm_self_attention': ('BiLSTM_SelfAttention', BiLSTMAttentionClassifier),
            'bilstm_multihead_attention': ('BiLSTM_MultiHead', BiLSTMAttentionClassifier),
        }
    
    @staticmethod
    def create_models(config: Dict) -> Dict[str, BaseNLPModel]:
        """
        Create all models defined in config.
        
        Args:
            config: Configuration dictionary containing model specifications
            
        Returns:
            Dictionary of model display name to model instance
        """
        models = {}
        
        # ML models
        if 'ml' in config.get('models', {}):
            ml_config = config['models']['ml']
            for config_key, (display_name, model_class) in NLPModelFactory.ML_MODEL_MAPPING.items():
                if config_key in ml_config:
                    params = ml_config[config_key].copy()
                    if 'random_state' not in params and config_key == 'logistic_regression':
                        params['random_state'] = RANDOM_SEED
                    models[display_name] = model_class(**params)
        
        # Standard LM models (fine-tuning)
        if 'lm' in config.get('models', {}):
            if not LM_AVAILABLE:
                print("Warning: LM models requested but transformers library not available.")
            else:
                lm_config = config['models']['lm']
                for config_key, (display_name, model_class) in NLPModelFactory.LM_MODEL_MAPPING.items():
                    if config_key in lm_config:
                        params = lm_config[config_key].copy()
                        params['random_state'] = params.get('random_state', RANDOM_SEED)
                        models[display_name] = model_class(**params)
        
        # Accelerate-based models
        if 'accelerate' in config.get('models', {}):
            if not ACCELERATE_AVAILABLE:
                print("Warning: Accelerate models requested but accelerate library not available.")
                print("Install with: pip install accelerate")
            else:
                accel_config = config['models']['accelerate']
                for config_key, (display_name, model_class) in NLPModelFactory.ACCELERATE_MODEL_MAPPING.items():
                    if config_key in accel_config:
                        params = accel_config[config_key].copy()
                        params['random_state'] = params.get('random_state', RANDOM_SEED)
                        models[display_name] = model_class(**params)
        
        # Unsloth LoRA models
        if 'unsloth' in config.get('models', {}):
            if not UNSLOTH_AVAILABLE:
                print("Warning: Unsloth models requested but unsloth library not available.")
                print("Install with: pip install unsloth trl")
            else:
                unsloth_config = config['models']['unsloth']
                for config_key, (display_name, model_class) in NLPModelFactory.UNSLOTH_MODEL_MAPPING.items():
                    if config_key in unsloth_config:
                        params = unsloth_config[config_key].copy()
                        params['random_state'] = params.get('random_state', RANDOM_SEED)
                        models[display_name] = model_class(**params)
        
        # Prompting models
        if 'prompting' in config.get('models', {}):
            if not PROMPTING_AVAILABLE:
                print("Warning: Prompting models requested but langchain library not available.")
            else:
                prompting_config = config['models']['prompting']
                for config_key in prompting_config.keys():
                    if config_key in NLPModelFactory.PROMPTING_MODEL_MAPPING:
                        display_name, model_class = NLPModelFactory.PROMPTING_MODEL_MAPPING[config_key]
                        params = prompting_config[config_key].copy()
                        params['random_state'] = params.get('random_state', RANDOM_SEED)
                        models[display_name] = model_class(**params)
        
        # DSpy models
        if 'dspy' in config.get('models', {}):
            if not DSPY_AVAILABLE:
                print("Warning: DSpy models requested but dspy-ai library not available.")
                print("Install with: pip install dspy-ai")
            else:
                dspy_config = config['models']['dspy']
                for config_key, (display_name, model_class) in NLPModelFactory.DSPY_MODEL_MAPPING.items():
                    if config_key in dspy_config:
                        params = dspy_config[config_key].copy()
                        params['random_state'] = params.get('random_state', RANDOM_SEED)
                        models[display_name] = model_class(**params)
        
        # Deep learning models (PyTorch)
        if 'deep' in config.get('models', {}):
            deep_config = config['models']['deep']
            
            # PyTorch models (MLP, CNN, Transformer)
            if DEEP_AVAILABLE:
                for config_key, (display_name, model_class) in NLPModelFactory.DEEP_MODEL_MAPPING.items():
                    if config_key in deep_config:
                        params = deep_config[config_key].copy()
                        params['random_state'] = params.get('random_state', RANDOM_SEED)
                        models[display_name] = model_class(**params)
            
            # Lightning models (LSTM, GRU, Attention)
            if DEEP_LIGHTNING_AVAILABLE:
                for config_key, (display_name, model_class) in NLPModelFactory.DEEP_LIGHTNING_MODEL_MAPPING.items():
                    if config_key in deep_config:
                        params = deep_config[config_key].copy()
                        params['random_state'] = params.get('random_state', RANDOM_SEED)
                        models[display_name] = model_class(**params)
            
            if not DEEP_AVAILABLE and not DEEP_LIGHTNING_AVAILABLE:
                print("Warning: Deep learning models requested but PyTorch not available.")
                print("Install with: pip install torch pytorch-lightning")
        
        return models
    
    @staticmethod
    def create_single_model(model_key: str, config: Dict) -> BaseNLPModel:
        """
        Create a single model by its config key.
        
        Args:
            model_key: Key identifying the model
            config: Configuration dictionary
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model key is not found or required libraries unavailable
        """
        # Check ML models
        if model_key in NLPModelFactory.ML_MODEL_MAPPING:
            display_name, model_class = NLPModelFactory.ML_MODEL_MAPPING[model_key]
            params = config.get('models', {}).get('ml', {}).get(model_key, {}).copy()
            if 'random_state' not in params and model_key == 'logistic_regression':
                params['random_state'] = RANDOM_SEED
            return model_class(**params)
        
        # Check LM models
        if model_key in NLPModelFactory.LM_MODEL_MAPPING:
            if not LM_AVAILABLE:
                raise ValueError(f"LM model '{model_key}' requires transformers. pip install transformers torch")
            display_name, model_class = NLPModelFactory.LM_MODEL_MAPPING[model_key]
            params = config.get('models', {}).get('lm', {}).get(model_key, {}).copy()
            params['random_state'] = params.get('random_state', RANDOM_SEED)
            return model_class(**params)
        
        # Check Accelerate models
        if model_key in NLPModelFactory.ACCELERATE_MODEL_MAPPING:
            if not ACCELERATE_AVAILABLE:
                raise ValueError(f"Accelerate model '{model_key}' requires accelerate. pip install accelerate")
            display_name, model_class = NLPModelFactory.ACCELERATE_MODEL_MAPPING[model_key]
            params = config.get('models', {}).get('accelerate', {}).get(model_key, {}).copy()
            params['random_state'] = params.get('random_state', RANDOM_SEED)
            return model_class(**params)
        
        # Check Unsloth models
        if model_key in NLPModelFactory.UNSLOTH_MODEL_MAPPING:
            if not UNSLOTH_AVAILABLE:
                raise ValueError(f"Unsloth model '{model_key}' requires unsloth. pip install unsloth trl")
            display_name, model_class = NLPModelFactory.UNSLOTH_MODEL_MAPPING[model_key]
            params = config.get('models', {}).get('unsloth', {}).get(model_key, {}).copy()
            params['random_state'] = params.get('random_state', RANDOM_SEED)
            return model_class(**params)
        
        # Check Prompting models
        if model_key in NLPModelFactory.PROMPTING_MODEL_MAPPING:
            if not PROMPTING_AVAILABLE:
                raise ValueError(f"Prompting model '{model_key}' requires langchain. pip install langchain huggingface_hub")
            display_name, model_class = NLPModelFactory.PROMPTING_MODEL_MAPPING[model_key]
            params = config.get('models', {}).get('prompting', {}).get(model_key, {}).copy()
            params['random_state'] = params.get('random_state', RANDOM_SEED)
            return model_class(**params)
        
        # Check DSpy models
        if model_key in NLPModelFactory.DSPY_MODEL_MAPPING:
            if not DSPY_AVAILABLE:
                raise ValueError(f"DSpy model '{model_key}' requires dspy-ai. pip install dspy-ai")
            display_name, model_class = NLPModelFactory.DSPY_MODEL_MAPPING[model_key]
            params = config.get('models', {}).get('dspy', {}).get(model_key, {}).copy()
            params['random_state'] = params.get('random_state', RANDOM_SEED)
            return model_class(**params)
        
        # Check Deep learning models (PyTorch)
        if model_key in NLPModelFactory.DEEP_MODEL_MAPPING:
            if not DEEP_AVAILABLE:
                raise ValueError(f"Deep model '{model_key}' requires torch. pip install torch")
            display_name, model_class = NLPModelFactory.DEEP_MODEL_MAPPING[model_key]
            params = config.get('models', {}).get('deep', {}).get(model_key, {}).copy()
            params['random_state'] = params.get('random_state', RANDOM_SEED)
            return model_class(**params)
        
        # Check Deep Lightning models (LSTM, GRU, Attention)
        if model_key in NLPModelFactory.DEEP_LIGHTNING_MODEL_MAPPING:
            if not DEEP_LIGHTNING_AVAILABLE:
                raise ValueError(f"Deep model '{model_key}' requires pytorch-lightning. pip install pytorch-lightning")
            display_name, model_class = NLPModelFactory.DEEP_LIGHTNING_MODEL_MAPPING[model_key]
            params = config.get('models', {}).get('deep', {}).get(model_key, {}).copy()
            params['random_state'] = params.get('random_state', RANDOM_SEED)
            return model_class(**params)
        
        # Build error message with all available models
        all_available = []
        all_available.extend(list(NLPModelFactory.ML_MODEL_MAPPING.keys()))
        if LM_AVAILABLE:
            all_available.extend(list(NLPModelFactory.LM_MODEL_MAPPING.keys()))
        if ACCELERATE_AVAILABLE:
            all_available.extend(list(NLPModelFactory.ACCELERATE_MODEL_MAPPING.keys()))
        if UNSLOTH_AVAILABLE:
            all_available.extend(list(NLPModelFactory.UNSLOTH_MODEL_MAPPING.keys()))
        if PROMPTING_AVAILABLE:
            all_available.extend(list(NLPModelFactory.PROMPTING_MODEL_MAPPING.keys()))
        if DSPY_AVAILABLE:
            all_available.extend(list(NLPModelFactory.DSPY_MODEL_MAPPING.keys()))
        if DEEP_AVAILABLE:
            all_available.extend(list(NLPModelFactory.DEEP_MODEL_MAPPING.keys()))
        if DEEP_LIGHTNING_AVAILABLE:
            all_available.extend(list(NLPModelFactory.DEEP_LIGHTNING_MODEL_MAPPING.keys()))
        
        raise ValueError(f"Unknown model key: '{model_key}'. Available models: {all_available}")
    
    @staticmethod
    def get_model_params(model: BaseNLPModel) -> Dict:
        """Get model parameters."""
        return model.get_params()
    
    @staticmethod
    def is_lm_model(model: BaseNLPModel) -> bool:
        """Check if a model requires raw text input (LM, prompting, deep, etc.)."""
        if not hasattr(model, 'model_type'):
            return False
        model_type = model.model_type
        return (model_type.startswith('lm') or 
                model_type.startswith('prompting') or
                model_type.startswith('deep') or
                'accelerate' in model_type or
                'unsloth' in model_type)
    
    @staticmethod
    def is_prompting_model(model: BaseNLPModel) -> bool:
        """Check if a model is a prompting model (no actual training)."""
        return hasattr(model, 'model_type') and 'prompting' in model.model_type
    
    @staticmethod
    def is_deep_model(model: BaseNLPModel) -> bool:
        """Check if a model is a deep learning model."""
        return hasattr(model, 'model_type') and model.model_type.startswith('deep')
    
    @staticmethod
    def get_available_models() -> Dict[str, list]:
        """Get all available model types."""
        deep_models = []
        if DEEP_AVAILABLE:
            deep_models.extend(list(NLPModelFactory.DEEP_MODEL_MAPPING.keys()))
        if DEEP_LIGHTNING_AVAILABLE:
            deep_models.extend(list(NLPModelFactory.DEEP_LIGHTNING_MODEL_MAPPING.keys()))
        
        return {
            'ml': list(NLPModelFactory.ML_MODEL_MAPPING.keys()),
            'deep': deep_models,
            'lm': list(NLPModelFactory.LM_MODEL_MAPPING.keys()) if LM_AVAILABLE else [],
            'accelerate': list(NLPModelFactory.ACCELERATE_MODEL_MAPPING.keys()) if ACCELERATE_AVAILABLE else [],
            'unsloth': list(NLPModelFactory.UNSLOTH_MODEL_MAPPING.keys()) if UNSLOTH_AVAILABLE else [],
            'prompting': list(NLPModelFactory.PROMPTING_MODEL_MAPPING.keys()) if PROMPTING_AVAILABLE else [],
            'dspy': list(NLPModelFactory.DSPY_MODEL_MAPPING.keys()) if DSPY_AVAILABLE else [],
        }
    
    @staticmethod
    def print_available_models():
        """Print all available models in a formatted way."""
        available = NLPModelFactory.get_available_models()
        print("\nAvailable Models:")
        print("=" * 50)
        for category, models in available.items():
            if models:
                print(f"\n{category.upper()}:")
                for model in models:
                    print(f"  - {model}")
            else:
                print(f"\n{category.upper()}: (not available - install dependencies)")
