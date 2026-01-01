"""
LangChain utilities for prompt-based sentiment classification.

Provides prompt templates, output parsers, and chain utilities for
zero-shot, few-shot, and chain-of-thought prompting.
"""
import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
import random


class SentimentOutputParser(BaseOutputParser[str]):
    """
    Parser for extracting sentiment labels from LLM outputs.
    
    Handles various output formats and extracts 'positive' or 'negative'.
    """
    
    def parse(self, text: str) -> str:
        """
        Parse the LLM output to extract sentiment label.
        
        Args:
            text: Raw LLM output text
            
        Returns:
            Normalized sentiment label ('positive' or 'negative')
        """
        # Clean and lowercase the text
        text_lower = text.lower().strip()
        
        # Look for explicit sentiment words
        # Check for positive indicators
        positive_patterns = [
            r'\bpositive\b',
            r'\bpos\b',
            r'sentiment[:\s]+positive',
            r'classification[:\s]+positive',
        ]
        
        negative_patterns = [
            r'\bnegative\b',
            r'\bneg\b',
            r'sentiment[:\s]+negative',
            r'classification[:\s]+negative',
        ]
        
        # Check positive patterns
        for pattern in positive_patterns:
            if re.search(pattern, text_lower):
                return 'positive'
        
        # Check negative patterns
        for pattern in negative_patterns:
            if re.search(pattern, text_lower):
                return 'negative'
        
        # If no clear match, look at the first word or line
        first_word = text_lower.split()[0] if text_lower.split() else ''
        if 'pos' in first_word:
            return 'positive'
        elif 'neg' in first_word:
            return 'negative'
        
        # Default fallback - check which word appears first
        pos_idx = text_lower.find('positive')
        neg_idx = text_lower.find('negative')
        
        if pos_idx == -1 and neg_idx == -1:
            # No sentiment word found, make best guess based on sentiment words
            positive_words = ['good', 'great', 'excellent', 'love', 'amazing', 'best', 'happy']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'disappointed']
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            return 'positive' if pos_count >= neg_count else 'negative'
        
        if pos_idx == -1:
            return 'negative'
        if neg_idx == -1:
            return 'positive'
        
        return 'positive' if pos_idx < neg_idx else 'negative'
    
    @property
    def _type(self) -> str:
        return "sentiment_output_parser"


class ChainOfThoughtParser(BaseOutputParser[Dict[str, Any]]):
    """
    Parser for chain-of-thought outputs that extracts reasoning and final answer.
    """
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse CoT output to extract reasoning steps and final sentiment.
        
        Args:
            text: Raw LLM output with reasoning
            
        Returns:
            Dictionary with 'reasoning' and 'sentiment' keys
        """
        text_lower = text.lower()
        
        # Extract the final sentiment
        sentiment_parser = SentimentOutputParser()
        
        # Try to find conclusion/final answer section
        conclusion_markers = [
            'final sentiment:', 'conclusion:', 'therefore:', 
            'overall sentiment:', 'final answer:', 'sentiment:'
        ]
        
        reasoning = text
        final_section = text
        
        for marker in conclusion_markers:
            if marker in text_lower:
                idx = text_lower.rfind(marker)
                final_section = text[idx:]
                reasoning = text[:idx]
                break
        
        sentiment = sentiment_parser.parse(final_section)
        
        return {
            'reasoning': reasoning.strip(),
            'sentiment': sentiment
        }
    
    @property
    def _type(self) -> str:
        return "cot_output_parser"


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

def get_zero_shot_template() -> PromptTemplate:
    """
    Get the zero-shot prompt template for sentiment classification.
    
    Returns:
        LangChain PromptTemplate for zero-shot classification
    """
    template = """Classify the sentiment of this product review.
Reply with exactly one word: positive or negative.

Review: {review}
Sentiment:"""
    
    return PromptTemplate(
        input_variables=["review"],
        template=template
    )


def get_zero_shot_template_instruct() -> PromptTemplate:
    """
    Get an instruction-tuned model compatible zero-shot template.
    
    Returns:
        LangChain PromptTemplate optimized for instruction-tuned models
    """
    template = """<s>[INST] You are a sentiment analysis assistant. Classify the following product review as either 'positive' or 'negative'. Respond with only one word.

Review: {review}

Sentiment: [/INST]"""
    
    return PromptTemplate(
        input_variables=["review"],
        template=template
    )


def get_few_shot_template(examples: List[Dict[str, str]] = None) -> FewShotPromptTemplate:
    """
    Get the few-shot prompt template with examples.
    
    Args:
        examples: List of example dictionaries with 'review' and 'sentiment' keys.
                 If None, uses default examples.
    
    Returns:
        LangChain FewShotPromptTemplate for few-shot classification
    """
    if examples is None:
        examples = get_default_examples()
    
    # Template for each example
    example_template = PromptTemplate(
        input_variables=["review", "sentiment"],
        template="Review: {review}\nSentiment: {sentiment}"
    )
    
    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix="Classify the sentiment of product reviews as positive or negative.\n\nExamples:\n",
        suffix="\nNow classify this review:\n\nReview: {review}\nSentiment:",
        input_variables=["review"]
    )


def get_few_shot_template_instruct(examples: List[Dict[str, str]] = None) -> PromptTemplate:
    """
    Get instruction-tuned model compatible few-shot template.
    
    Args:
        examples: List of example dictionaries
        
    Returns:
        PromptTemplate with examples embedded for instruction models
    """
    if examples is None:
        examples = get_default_examples()
    
    # Build examples string
    examples_str = "\n".join([
        f"Review: {ex['review']}\nSentiment: {ex['sentiment']}"
        for ex in examples
    ])
    
    template = f"""<s>[INST] You are a sentiment analysis assistant. Classify product reviews as 'positive' or 'negative'.

Here are some examples:

{examples_str}

Now classify this review. Respond with only one word (positive or negative).

Review: {{review}}

Sentiment: [/INST]"""
    
    return PromptTemplate(
        input_variables=["review"],
        template=template
    )


def get_chain_of_thought_template() -> PromptTemplate:
    """
    Get the chain-of-thought prompt template.
    
    Returns:
        LangChain PromptTemplate for CoT reasoning
    """
    template = """Analyze the sentiment of this product review step by step.

Review: {review}

Let's analyze this step by step:

Step 1 - Identify key sentiment-bearing words and phrases:

Step 2 - Analyze the overall tone (positive signals vs negative signals):

Step 3 - Consider any nuances or mixed sentiments:

Final Sentiment (positive or negative):"""
    
    return PromptTemplate(
        input_variables=["review"],
        template=template
    )


def get_chain_of_thought_template_instruct() -> PromptTemplate:
    """
    Get instruction-tuned model compatible CoT template.
    
    Returns:
        PromptTemplate for CoT with instruction format
    """
    template = """<s>[INST] You are a sentiment analysis assistant. Analyze this product review step by step, then provide a final sentiment classification.

Review: {review}

Please follow these steps:
1. Identify key sentiment-bearing words and phrases
2. Analyze the overall tone (count positive vs negative signals)
3. Consider any nuances or mixed sentiments
4. Provide your final sentiment classification (positive or negative)

Begin your analysis: [/INST]

Step 1 - Key sentiment words:"""
    
    return PromptTemplate(
        input_variables=["review"],
        template=template
    )


def get_default_examples() -> List[Dict[str, str]]:
    """
    Get default few-shot examples for sentiment classification.
    
    Returns:
        List of example dictionaries with balanced positive/negative examples
    """
    return [
        {
            "review": "This product exceeded my expectations! Great quality and fast shipping.",
            "sentiment": "positive"
        },
        {
            "review": "Terrible quality. Broke after one week of use. Complete waste of money.",
            "sentiment": "negative"
        },
        {
            "review": "Absolutely love it! Best purchase I've made this year.",
            "sentiment": "positive"
        },
        {
            "review": "Very disappointed. The product looks nothing like the pictures.",
            "sentiment": "negative"
        },
        {
            "review": "Works perfectly and arrived earlier than expected. Highly recommend!",
            "sentiment": "positive"
        },
        {
            "review": "Poor customer service and the item was damaged. Would not buy again.",
            "sentiment": "negative"
        }
    ]


def select_examples(
    examples: List[Dict[str, str]],
    num_examples: int = 3,
    balanced: bool = True
) -> List[Dict[str, str]]:
    """
    Select a subset of examples for few-shot prompting.
    
    Args:
        examples: Full list of available examples
        num_examples: Number of examples to select
        balanced: Whether to balance positive/negative examples
        
    Returns:
        Selected subset of examples
    """
    if not balanced:
        return random.sample(examples, min(num_examples, len(examples)))
    
    # Separate by sentiment
    positive = [ex for ex in examples if ex['sentiment'] == 'positive']
    negative = [ex for ex in examples if ex['sentiment'] == 'negative']
    
    # Select balanced samples
    num_each = num_examples // 2
    remainder = num_examples % 2
    
    selected = []
    selected.extend(random.sample(positive, min(num_each + remainder, len(positive))))
    selected.extend(random.sample(negative, min(num_each, len(negative))))
    
    # Shuffle to mix positive/negative
    random.shuffle(selected)
    
    return selected


# Output parser instances
sentiment_parser = SentimentOutputParser()
cot_parser = ChainOfThoughtParser()



