"""
LLM Prompting module for sentiment classification.

Provides:
1. Zero-shot, few-shot, and chain-of-thought prompting (LangChain)
2. DSpy for learnable prompt optimization

Example usage - LangChain Prompting:
    from models.LMs.prompting import ZeroShotPromptClassifier, FewShotPromptClassifier
    
    # Zero-shot classification using HuggingFace API
    classifier = ZeroShotPromptClassifier(
        backend="hf_api",
        model_name="mistralai/Mistral-7B-Instruct-v0.2"
    )
    predictions = classifier.predict(texts)
    
    # Few-shot classification with examples from training data
    classifier = FewShotPromptClassifier(num_examples=3)
    classifier.fit(train_texts, train_labels)  # Selects examples
    predictions = classifier.predict(test_texts)

Example usage - DSpy Prompt Optimization:
    from models.LMs.prompting import DSPyClassifier
    
    # Optimized prompting with chain-of-thought
    classifier = DSPyClassifier(
        technique="cot",
        optimize=True,
        optimizer="bootstrap"
    )
    classifier.fit(train_texts, train_labels)  # Optimizes prompts
    predictions = classifier.predict(test_texts)

Note: Requires HuggingFace API token for API-based inference.
Set the HF_TOKEN environment variable or pass token to the backend.
"""

from models.LMs.prompting.base_prompt import BasePromptClassifier
from models.LMs.prompting.classifiers import (
    ZeroShotPromptClassifier,
    FewShotPromptClassifier,
    ChainOfThoughtClassifier,
    # Aliases
    ZeroShot,
    FewShot,
    CoT,
)
from models.LMs.prompting.inference import (
    BaseInferenceBackend,
    HFAPIBackend,
    LocalHFBackend,
    get_backend,
)
from models.LMs.prompting.langchain_utils import (
    SentimentOutputParser,
    ChainOfThoughtParser,
    get_zero_shot_template,
    get_few_shot_template,
    get_chain_of_thought_template,
    get_default_examples,
    select_examples,
)

# DSpy exports - wrapped in try/except
DSPY_AVAILABLE = False
try:
    from models.LMs.prompting.dspy_optimizer import (
        DSPyClassifier,
        DSPyReActClassifier,
        DSPyPredict,
        DSPyCoT,
        DSPyOptimized,
    )
    DSPY_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    # Base class
    'BasePromptClassifier',
    
    # LangChain Classifier implementations
    'ZeroShotPromptClassifier',
    'FewShotPromptClassifier', 
    'ChainOfThoughtClassifier',
    
    # Aliases
    'ZeroShot',
    'FewShot',
    'CoT',
    
    # Inference backends
    'BaseInferenceBackend',
    'HFAPIBackend',
    'LocalHFBackend',
    'get_backend',
    
    # LangChain utilities
    'SentimentOutputParser',
    'ChainOfThoughtParser',
    'get_zero_shot_template',
    'get_few_shot_template',
    'get_chain_of_thought_template',
    'get_default_examples',
    'select_examples',
    
    # Availability flag
    'DSPY_AVAILABLE',
]

# Add DSpy exports if available
if DSPY_AVAILABLE:
    __all__.extend([
        'DSPyClassifier',
        'DSPyReActClassifier',
        'DSPyPredict',
        'DSPyCoT',
        'DSPyOptimized',
    ])
