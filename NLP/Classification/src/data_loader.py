"""
NLP data loading and preprocessing module
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from typing import Dict, List
import os
import json
import glob
from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
import string
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data (only if not already present)
try:
    import nltk
    from nltk.data import find
    
    def safe_download_nltk(resource_name: str, quiet: bool = True):
        """
        Safely download NLTK resource only if it doesn't already exist.
        nltk.download() already checks for existence, but we add extra safety.
        
        Args:
            resource_name: Name of the NLTK resource to download
            quiet: Whether to suppress output
        """
        try:
            # Check if resource already exists
            if resource_name == 'punkt_tab':
                # Special handling for punkt_tab (newer NLTK versions)
                try:
                    find('tokenizers/punkt_tab')
                    return  # Already exists
                except LookupError:
                    pass
            else:
                try:
                    find(resource_name)
                    return  # Already exists
                except LookupError:
                    pass
            
            # Resource doesn't exist, download it
            # nltk.download() will skip if already exists, but we check first for efficiency
            nltk.download(resource_name, quiet=quiet)
        except Exception:
            # If anything fails, try downloading anyway (nltk.download handles duplicates)
            try:
                nltk.download(resource_name, quiet=quiet)
            except Exception:
                pass
    
    # Download required resources
    # Try punkt_tab first (newer NLTK versions), fallback to punkt if needed
    try:
        safe_download_nltk('punkt_tab', quiet=True)
    except Exception:
        # Fallback to punkt for older NLTK versions
        try:
            safe_download_nltk('punkt', quiet=True)
        except Exception:
            pass
    
    # Download other required resources
    safe_download_nltk('stopwords', quiet=True)
    safe_download_nltk('wordnet', quiet=True)
    
except Exception:
    # If NLTK is not available or download fails, continue anyway
    # The error will be caught later when trying to use NLTK functions
    pass


class NLPDataLoader:
    """Data loader for NLP classification tasks."""
    
    def __init__(self, config: Dict):
        """
        Initialize the NLP data loader.
        
        Args:
            config: Configuration dictionary containing data parameters
        """
        self.config = config
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize stemmer if needed
        self.stemmer = None
        if config.get('preprocessing', {}).get('stemming') == 'porter':
            self.stemmer = PorterStemmer()
        elif config.get('preprocessing', {}).get('stemming') == 'snowball':
            self.stemmer = SnowballStemmer('english')
    
    def read_from_lxml(self, file_paths: List[str]) -> List[str]:
        """
        Read reviews from XML files using BeautifulSoup.
        
        Args:
            file_paths: List of file paths to read from
            
        Returns:
            List of review texts
        """
        review_list = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = bs(f, 'lxml')
                    reviews = soup.find_all('review_text')
                    for review in reviews:
                        review_list.append(review.get_text())
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
        
        return review_list
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load data from XML review files.
        
        Args:
            data_path: Path pattern for review files (if None, uses config)
            
        Returns:
            DataFrame with 'review' and 'sentiment' columns
        """
        if data_path is None:
            data_path = self.config['data'].get('data_path', 'sorted_data_acl/**/')
        
        # Normalize path separators and handle glob patterns
        # Convert backslashes to forward slashes for glob (works on Windows too)
        normalized_path = data_path.replace('\\', '/')
        
        # Ensure the pattern ends properly for recursive search
        if normalized_path.endswith('**'):
            # Path ends with **, add trailing slash
            pattern_base = normalized_path + '/'
        elif normalized_path.endswith('**/'):
            # Path ends with **/, use as-is
            pattern_base = normalized_path
        elif '**' not in normalized_path:
            # No ** pattern, add it for recursive search
            pattern_base = normalized_path.rstrip('/') + '/**/'
        else:
            # ** is in the middle, use as-is
            pattern_base = normalized_path
        
        # Construct glob patterns
        positive_pattern = pattern_base + 'positive.*'
        negative_pattern = pattern_base + 'negative.*'
        
        # Find positive and negative review files
        positive_files = glob.glob(positive_pattern, recursive=True)
        negative_files = glob.glob(negative_pattern, recursive=True)
        
        if not positive_files or not negative_files:
            raise ValueError(f"No review files found at path: {data_path}\n"
                           f"  Positive files found: {len(positive_files)}\n"
                           f"  Negative files found: {len(negative_files)}\n"
                           f"  Tried patterns:\n"
                           f"    Positive: {positive_pattern}\n"
                           f"    Negative: {negative_pattern}\n"
                           f"  Please check that:\n"
                           f"    1. The path exists: {os.path.dirname(data_path) if os.path.dirname(data_path) else data_path}\n"
                           f"    2. Review files exist in subdirectories")
        
        print(f"Found {len(positive_files)} positive review files")
        print(f"Found {len(negative_files)} negative review files")
        
        # Read reviews
        positive_reviews = self.read_from_lxml(positive_files)
        negative_reviews = self.read_from_lxml(negative_files)
        
        # Create labels (1 for positive, 0 for negative)
        labels = np.concatenate([
            np.ones(len(positive_reviews)),
            np.zeros(len(negative_reviews))
        ])
        
        # Combine reviews
        all_reviews = positive_reviews + negative_reviews
        
        # Create DataFrame
        df = pd.DataFrame({
            'review': all_reviews,
            'sentiment': labels
        })
        
        print(f"Loaded {len(df)} reviews total")
        print(f"  Positive: {len(positive_reviews)}")
        print(f"  Negative: {len(negative_reviews)}")
        
        return df
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text string.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        preprocess_config = self.config.get('preprocessing', {})
        
        # Lowercase
        if preprocess_config.get('lowercase', True):
            text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Process tokens
        processed_tokens = []
        for token in tokens:
            # Remove punctuation
            if preprocess_config.get('remove_punctuation', True):
                token = token.translate(str.maketrans('', '', string.punctuation))
            
            # Remove numbers
            if preprocess_config.get('remove_numbers', True):
                if token.isdigit():
                    continue
            
            # Keep only alphabetic tokens
            if not token.isalpha():
                continue
            
            # Remove stopwords
            if preprocess_config.get('remove_stopwords', True):
                if token in self.stop_words:
                    continue
            
            # Stemming
            if self.stemmer is not None:
                token = self.stemmer.stem(token)
            
            # Lemmatization
            if preprocess_config.get('lemmatization', True):
                token = self.lemmatizer.lemmatize(token)
            
            if token:  # Only add non-empty tokens
                processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]
    
    def create_vectorizer(self) -> None:
        """
        Create vectorizer based on configuration.
        """
        preprocess_config = self.config.get('preprocessing', {})
        method = preprocess_config.get('vectorization_method', 'tfidf')
        
        params = {
            'max_features': preprocess_config.get('max_features', 10000),
            'min_df': preprocess_config.get('min_df', 2),
            'max_df': preprocess_config.get('max_df', 0.95),
            'ngram_range': tuple(preprocess_config.get('ngram_range', [1, 2]))
        }
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(**params)
        elif method == 'count':
            self.vectorizer = CountVectorizer(**params)
        elif method == 'hashing':
            self.vectorizer = HashingVectorizer(**params)
        else:
            raise ValueError(f"Unknown vectorization method: {method}")
    
    def vectorize_texts(self, texts: List[str], fit: bool = True):
        """
        Vectorize texts using the configured vectorizer.
        
        Args:
            texts: List of texts to vectorize
            fit: Whether to fit the vectorizer (True for training data)
            
        Returns:
            Vectorized features (sparse matrix)
        """
        if self.vectorizer is None:
            self.create_vectorizer()
        
        if fit:
            return self.vectorizer.fit_transform(texts)
        else:
            return self.vectorizer.transform(texts)
    
    def train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = None,
        val_size: float = None,
        random_state: int = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            random_state: Random seed
            
        Returns:
            Dictionary containing train, validation, and test DataFrames
        """
        if test_size is None:
            test_size = self.config['data'].get('test_size', 0.2)
        if val_size is None:
            val_size = self.config['data'].get('validation_size', 0.1)
        if random_state is None:
            random_state = self.config['data'].get('random_state', 42)
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['sentiment']
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=random_state,
            stratify=train_val_df['sentiment']
        )
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def prepare_data(
        self,
        data_path: str = None,
        preprocess: bool = True
    ) -> Dict:
        """
        Complete data preparation pipeline for NLP classification.
        
        Args:
            data_path: Path pattern for review files (if None, uses config)
            preprocess: Whether to preprocess texts
            
        Returns:
            Dictionary containing processed train/val/test data and vectorizers
        """
        # Load data
        print("Loading data...")
        df = self.load_data(data_path)
        
        # Preprocess texts
        if preprocess:
            print("Preprocessing texts...")
            df['review_processed'] = self.preprocess_texts(df['review'].tolist())
        else:
            df['review_processed'] = df['review']
        
        # Split data
        print("Splitting data...")
        splits = self.train_test_split(df)
        
        # Vectorize texts
        print("Vectorizing texts...")
        X_train = self.vectorize_texts(
            splits['train']['review_processed'].tolist(),
            fit=True
        )
        X_val = self.vectorize_texts(
            splits['val']['review_processed'].tolist(),
            fit=False
        )
        X_test = self.vectorize_texts(
            splits['test']['review_processed'].tolist(),
            fit=False
        )
        
        # Extract labels
        y_train = splits['train']['sentiment'].values
        y_val = splits['val']['sentiment'].values
        y_test = splits['test']['sentiment'].values
        
        print(f"\nData preparation complete!")
        print(f"  Train size: {len(y_train)}")
        print(f"  Validation size: {len(y_val)}")
        print(f"  Test size: {len(y_test)}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'train_df': splits['train'],
            'val_df': splits['val'],
            'test_df': splits['test'],
            'vectorizer': self.vectorizer
        }
    
    def save_processed_data(self, data_dict: Dict, path: str) -> None:
        """
        Save processed data to disk.
        
        Args:
            data_dict: Dictionary of processed data
            path: Directory path to save data
        """
        os.makedirs(path, exist_ok=True)
        
        # Save metadata
        metadata = {
            'splits': ['train', 'val', 'test'],
            'saved_at': pd.Timestamp.now().isoformat(),
            'feature_dim': data_dict['X_train'].shape[1]
        }
        
        # Save metadata as JSON
        metadata_path = os.path.join(path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved metadata to {metadata_path}")

