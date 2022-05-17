import csv
import numpy as np
import pandas as pd
import spacy


nlp = spacy.load("en_core_web_sm")


"""
 Course:      Statistical Language processing - SS2022
 Assignment:  (A1)
 Author(s):   (Markus Schoch, Leixin)

 Honor Code:  I pledge that this program represents my/our own work.
"""


def load_data(filename, col_index):
    """1.1.1 Load the file into a list of sentences.
    Skips the headers and the chapter headings

    Parameters
    ----------
    filename - path to the file
    col_ind - index of   the column with English text

    Returns
    -------
    lst[lst[str]] - list of sentences
    """
    file = open(filename, encoding="UTF-8")
    tsv_file = csv.reader(file, delimiter="\t")
    next(tsv_file) #skip the first two rows (Head and author)
    next(tsv_file)
    en_list = list()
    for line in tsv_file:
        if 'Chapter' not in line[col_index]:    #skip the crow contains "CHAPTER"
            en_list.append(line[col_index])
    file.close()
    return en_list

def preprocess_sentence(sentence):
    """1.1.2 Takes a sentence as input and does the following preprocessing steps:
          1) punctuation removal
          2) lowercasing
          3) stopword removal
          4) tokenization
          5) lemmatization

    Parameters
    ----------
    sentence - the sentence to preprocess

    Returns
    -------
    lst[str] - list of strings where each string corresponds to a
    word lemma after all the previous preprocessing steps
    """

    doc = nlp(sentence.lower())
    text_no_punct = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return text_no_punct

def co_occurrence_matrix(dataset):
    """1.2 Constructs a co-occurrence matrix from the dataset.
    The length of the context window is fixed to 4.
    The matrix must be alphabetically sorted.

    Parameters
    ----------
    dataset - list of preprocessed sentences

    Returns
    -------
    VxV numpy array - where V is the length of the Vocabulary and
    each field of the matrix represents the co-occurrence counts of
    the appropriate word-context pair
    """
    pass


def ppmi_matrix(cooc_matrix, smoothing=None):
    """2.1 Constructs a PPMI matrix from the co-occurrence matrix.
    Also has a smoothing parameter that you are asked to
    implement in task 2.4

    Parameters
    ----------
    cooc_matrix - co-occurrence matrix
    smoothing - smoothing parameter

    Returns
    -------
    VxV numpy array - where V is the length of the Vocabulary and
    each field of the matrix represents the PPMI value of
    the appropriate word-context pair
    """
    pass


def ppmi(word, context, ppmi_mtx, dataset):
    """2.2 Obtains the PPMI value for a given word-context pair from the PPMI matrix.
    The dataset parameter allows you to get access to words

    Parameters
    ----------
    word - word from the word-context pair for which to obtain the PPMI value
    context - context from the word-context pair for which to obtain the PPMI value
    ppmi_mtx - PPMI matrix
    dataset - list of preprocessed sentences

    Returns
    -------
    float - PPMI value
    """
    pass


def get_word_vectors(ppmi_mtx, dataset):
    """2.3 Given a PPMI matrix and the dataset obtain the word vectors.

    Parameters
    ----------
    ppmi_mtx - PPMI matrix
    dataset - list of preprocessed sentences

    Returns
    -------
    dict - dictionary with words (strings) as keys and corresponding
    word vectors as values
    """
    pass


def k_most_similar(word, k, word_vectors):
    """2.3 Returns k most similar words to the input word.
    Uses cosine similarity to calculate word distance

    Parameters
    ----------
    word - target word
    k - number of most similar words to return
    word_vectors - dictionary of word vectors as returned by get_word_vectors()

    Returns
    -------
    lst[str] - list of k most similar words to the input word
    """
    pass


if __name__ == '__main__':
    # You can use this space to try out your code
    # and make sure it works as you'd expect
    # (this part is not graded)

    print(preprocess_sentence(sentence="This is a Tests Sentences and we're going hiking."))
#    for token in preprocess_sentence(sentence="This is a Tests Sentences and we're going hiking."):
#        print(token.lemma_)
