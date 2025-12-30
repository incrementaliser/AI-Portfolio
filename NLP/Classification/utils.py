"""This module contains the code for extracting data from .tr.gz files and also importing data for further use.

TO-DO:
1. Improve variables' names.
2. Add the capability of choosing paths.
3. Add docstring to function read_from_lxml.
4. Check if writing main() helps.
5. Check docstrings quality.
6. Add function shuffle_data_and_labels.
7. Save extracted data from lxml files locally and add a function to simply read them.
8. read about type hints + PEP 483, 484, and below links, and "->" operator
    * https://docs.python.org/3/library/typing.html
    * https://stackoverflow.com/questions/38727520/how-to-add-default-parameter-to-functions-when-using-type-hint
    * https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments
"""
import glob
from bs4 import BeautifulSoup as bs
import numpy as np
import pandas as pd


def read_from_lxml(directory):

    review_list = []

    for dir_ in directory:
        f = open(dir_, 'r')
        soup = bs(f, 'lxml')
        reviews = soup.find_all('review_text')
        for review in reviews:
            review_list.append(review.get_text())

    return review_list


def load_data(path, as_df=True):
    """ ATTENTION: You first need to have data extracted in order to import it.
    Use extract_data() before calling this method if you have not done already.
    
    For parameter `path`, only pass the address of 
    This method receives a path, and extracts data in that path.
    
    Parameters
    ----------
    as_df: bool, optional
        If True, the method returns a pandas DataFrame. Otherwise, it returns lists and numpy arrays.
    path: string, optional
        The path you wish to read data from. The default value is the current local path.

    Returns
    -------
    pos_revs: list of strings
        A list containing all positive reviews.
    neg_revs: list of strings
        A list containing all negative reviews.
    labels: numpy array
        A 2D array containing the labels of our data (1 for positive, and 0 for negative).
        By 2D I mean of shape (n,1).
    """

    # Reading from extracted files
    positive_reviews = glob.glob(path+'positive.*', recursive=True)
    negative_reviews = glob.glob(path+'negative.*', recursive=True)
    
    pos_revs = read_from_lxml(positive_reviews)
    neg_revs = read_from_lxml(negative_reviews)

    labels = generate_labels(len(pos_revs))
    if as_df:
        return pd.DataFrame({'review': pos_revs + neg_revs, 'sentiment': labels.flatten()})
    return pos_revs, neg_revs, labels


def generate_labels(length):
    """ Creates the labels for the data.

    Parameters
    ----------
    length: int
        The length of all reviews of the same type.
    
    Returns
    -------
    labels: numpy array
        A 2D array containing the labels of our data (1 for positive, and 0 for negative).
        By 2D I mean of shape (n,1).
    """
    labels = np.concatenate([np.ones(length), np.zeros(length)])
    labels = labels.reshape((-1, 1))
    return labels


def extract_data():
    """
    This method extracts data from .tr.gz compressed files to X files.
    """
    # path = input(" Please enter a valid path to extract data into, and read data from: ")
    pass

# def main():
# user_path = input(" Please enter the location of `sorted_data_acl` folder: ") # add an example.
