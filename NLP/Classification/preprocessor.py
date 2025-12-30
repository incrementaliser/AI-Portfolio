""" This  module helps with preprocessing text.

To-Do:
* Add the capability of choosing what to do and what not to do.
* Deciding between libraries to use.
* decide on the order of preprocessing tasks.
* Add capability of preprocessing on both sting and list of strings
"""

# -------------------- Importing necessary libraries --------------------
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords # which one, this or from sklearn? !!
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer #, WordNetLemmatizer
from tqdm import tqdm

# import string

# -------------------- Caching useful stuff --------------------
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()

# -------------------- function preprocess --------------------
def preprocess(raw_text):
    """
    Applies some text preprocessing on the input.
    Preprocessing tasks that will be applied are:
    * Tokenization
    * Convertion to lower-case
    * Only-alphabet filtering (Removal of punctuations,
        numbers, whitespaces, special characters, etc.)
    * Stopwords removal
    * Stemming
    
    Parameters
    ----------
    raw_text: string
        The input text to be processed.
    Returns
    -------
    processed_text: string
        The text after preprocessing.
    """

    tokenized = word_tokenize(raw_text)
    lower_cased = [word.lower() for word in tokenized]
    processed_text = [stemmer.stem(word) for word in lower_cased if word.isalpha() and word not in stop_words]

    return processed_text


# Cooool! Did I just wrote my first test case?
# def main():
#     data = ["arash is a good boy.", "sam is a good girl."]
#     proc = [preprocess(x) for x in data]

#     print(proc)

# if __name__ == "__main__":
#     main()