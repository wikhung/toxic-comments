from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

import re
import numpy as np


# Function to clean the comments
def text_processing(text, lower=True, keep_sw=False, regex=None):
    """
    :param text: pandas series; containing the comments
    :param lower: boolean; specify whether to lower the cases
    :param keep_sw: boolean; specify whether to keep English stopwords
    :param regex: string; specify the RE used to remove characters in the text
    :return: cleaned text
    """
    # Fill in NA for any missing comments
    txt = text.fillna("na")

    # Lower all cases
    if lower:
        print('Lowered the string')
        txt = txt.apply(lambda s: s.lower())

    # Replace \n by a space
    txt = txt.apply(lambda s: re.sub(r"\n", " ", s))

    # Remove English stopwords
    if not keep_sw:
        print("Remove English stopwords")
        sw = set(stopwords.words('english'))
        txt = txt.apply(lambda s: " ".join([word for word in s.split(" ") if word not in sw]))

    # Tokenize the sentences
    print('Tokenize the sentences')
    txt = txt.apply(lambda s: sent_tokenize(s))

    # Remove characters specified in RE
    if regex:
        print("Remove non-alphanumeric characters")
        txt = txt.apply(lambda s: [re.sub(regex, "", sent) for sent in s])

    return txt

# Function to build char2idx and idx2char dict
def build_idx(text):
    """
    :param text: pandas series; comments
    :return:
        num_chars: number of unique characters in the data
        char2idx: dict to convert character to idx
        idx2char: dict to convert idx to character
    """
    all_chars = set("".join([word for sent in text for word in sent]))
    num_chars = len(all_chars)

    char2idx = {ch: i + 1 for i, ch in enumerate(all_chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}

    return num_chars, char2idx, idx2char

def comments_to_idx(comments, nb_sent, max_len, char2idx):
    """
    :param comments: pandas series; comments
    :param nb_sent: int; maximum number of sentences in each comment
    :param max_len: int; maximum length of each sentence
    :param char2idx: dict; character to idx dictionary
    :return:
        processed_comments: numpy matrix; comments converted to index
                            with dim (len(comments), nb_sent, max_len)
    """

    # Initialize an empty matrix with specified dimensions
    processed_comments = np.zeros((len(comments), nb_sent, max_len))
    # Loop through the comments by sentences and characters
    # Store the conversion if they are within specified length
    for i, comment in enumerate(comments):
        for j, sent in enumerate(comment):
            if j < nb_sent:
                for t, char in enumerate(sent):
                    if t < max_len:
                        processed_comments[i, j, t] = char2idx[char]
    return processed_comments

class augmentation(object):
    def __init__(self, replace_set):
        """
        :param replace_set: list; characters that will be used to modify the input strings
        """
        self.replace_set = replace_set

    def add_character(self, s, choice):
        """
        :param s: string;
        :param choice: int; index to add character
        :return:
            s: string; modified string
        """
        replace_character = np.random.choice(self.replace_set, size=1)[0]
        s = s[:choice] + replace_character + s[choice:]
        return s

    def drop_character(self, s, choice):
        """
        :param s: string;
        :param choice: int; index to remove character
        :return:
            s: string; modified string
        """
        s = s[:choice] + s[choice + 1:]

        return s

    def replace_character(self, s, choice):
        """
        :param s: string;
        :param choice: int; index of character to replace
        :return:
            s: string; modified string
        """
        replace_character = np.random.choice(self.replace_set, size=1)[0]
        s = s[:choice] + replace_character + s[choice + 1:]

        return s

    def data_augmentation(self, s, prob):
        """
        :param prob: float; permutation probability
        :param aug_methods: dict; specifying the methods used for permutation
        :return:
            s; string
        """
        aug_methods = dict(zip(['add', 'replace', 'drop'],
                               [self.add_character, self.replace_character, self.drop_character]))

        rolls = np.random.choice([0, 1], size=len(s), p=[1 - prob, prob])
        choices = np.where(rolls == 1)[0]
        methods = np.random.choice(list(aug_methods.keys()), size=len(choices))
        s_len = len(s)

        for i, choice in enumerate(choices):
            chr_cnt = len(s) - s_len
            s = aug_methods[methods[i]](s, choice + chr_cnt, prob)

        return s