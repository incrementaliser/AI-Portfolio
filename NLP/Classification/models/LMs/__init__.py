"""
Language Model (LM) classifiers for NLP classification.

This module provides:
1. Transformer-based models (BERT, DistilBERT, RoBERTa) - Standard fine-tuning
2. Accelerate-based training - Multi-GPU, mixed precision
3. Unsloth LoRA - Efficient LLM fine-tuning (2-5x faster)
4. Prompt-based models - Zero-shot, few-shot, chain-of-thought
5. DSpy - Learnable prompt optimization

Example usage - Standard Fine-tuning:
    from models.LMs import BERTClassifier, DistilBERTClassifier
    
    model = DistilBERTClassifier(max_length=256, batch_size=32, epochs=3)
    model.fit(train_texts, train_labels, val_texts, val_labels)
    predictions = model.predict(test_texts)

Example usage - Accelerate (Multi-GPU):
    from models.LMs import BERTAccelerate, DistilBERTAccelerate
    
    model = BERTAccelerate(mixed_precision="fp16", gradient_accumulation_steps=2)
    model.fit(train_texts, train_labels)

Example usage - Unsloth LoRA (Efficient LLM fine-tuning):
    from models.LMs import UnslothLoRAClassifier, LlamaLoRA
    
    model = LlamaLoRA(lora_r=16, load_in_4bit=True)
    model.fit(train_texts, train_labels)

Example usage - Prompting:
    from models.LMs import ZeroShotPromptClassifier, FewShotPromptClassifier
    
    model = ZeroShotPromptClassifier(backend="hf_api")
    predictions = model.predict(texts)

Example usage - DSpy (Prompt Optimization):
    from models.LMs import DSPyClassifier
    
    model = DSPyClassifier(technique="cot", optimize=True)
    model.fit(train_texts, train_labels)
    predictions = model.predict(test_texts)
"""

# Standard fine-tuning models
from models.LMs.base_lm import BaseLMModel
from models.LMs.bert_models import (
    BERTClassifier,
    DistilBERTClassifier,
    RoBERTaClassifier
)
from models.LMs.lm_data import (
    TextClassificationDataset,
    LMDataModule
)

# Accelerate-based models - wrapped in try/except
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

# Unsloth LoRA models - wrapped in try/except
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

# Prompting models - wrapped in try/except
PROMPTING_AVAILABLE = False
try:
    from models.LMs.prompting import (
        BasePromptClassifier,
        ZeroShotPromptClassifier,
        FewShotPromptClassifier,
        ChainOfThoughtClassifier,
        ZeroShot,
        FewShot,
        CoT,
        HFAPIBackend,
        LocalHFBackend,
    )
    PROMPTING_AVAILABLE = True
except ImportError:
    pass

# DSpy models - wrapped in try/except
DSPY_AVAILABLE = False
try:
    from models.LMs.prompting.dspy_optimizer import (
        DSPyClassifier,
        DSPyReActClassifier,
        DSPyPredict,
        DSPyCoT,
        DSPyOptimized
    )
    DSPY_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    # Base classes
    'BaseLMModel',
    
    # Standard fine-tuning models
    'BERTClassifier',
    'DistilBERTClassifier',
    'RoBERTaClassifier',
    
    # Data utilities
    'TextClassificationDataset',
    'LMDataModule',
    
    # Availability flags
    'ACCELERATE_AVAILABLE',
    'UNSLOTH_AVAILABLE',
    'PROMPTING_AVAILABLE',
    'DSPY_AVAILABLE',
]

# Add Accelerate exports if available
if ACCELERATE_AVAILABLE:
    __all__.extend([
        'AccelerateTrainer',
        'BERTAccelerate',
        'DistilBERTAccelerate',
        'RoBERTaAccelerate',
    ])

# Add Unsloth exports if available
if UNSLOTH_AVAILABLE:
    __all__.extend([
        'UnslothLoRAClassifier',
        'LlamaLoRA',
        'PhiLoRA',
        'MistralLoRA',
    ])

# Add prompting exports if available
if PROMPTING_AVAILABLE:
    __all__.extend([
        'BasePromptClassifier',
        'ZeroShotPromptClassifier',
        'FewShotPromptClassifier',
        'ChainOfThoughtClassifier',
        'ZeroShot',
        'FewShot',
        'CoT',
        'HFAPIBackend',
        'LocalHFBackend',
    ])

# Add DSpy exports if available
if DSPY_AVAILABLE:
    __all__.extend([
        'DSPyClassifier',
        'DSPyReActClassifier',
        'DSPyPredict',
        'DSPyCoT',
        'DSPyOptimized',
    ])
