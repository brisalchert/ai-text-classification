#-----------------------------------------------------------------------------------------------------------------------
# AI Text Classification
#
# Python script for training a CNN for classifying AI-generated versus human-generated text.
#-----------------------------------------------------------------------------------------------------------------------

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import re

# Load dataset from csv
ai_human_df = pd.read_csv('ai_human.csv')

# Preview data
print(ai_human_df.head())
print(ai_human_df.info())

# Visualize distribution of data
sns.countplot(x=ai_human_df['generated'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

def get_tokenizer():
    """
    Gets a tokenizer function for basic english sentences. Adapted from the deprecated TorchText library.
    :return: a tokenizer function for basic english splitting on whitespace.
    """

    _patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]

    _replacements = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]

    _patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))

    def _basic_english_normalize(line):
        """
        Basic normalization for a line of text in english.
        :param line: the text to normalize.
        :return: a list of tokens splitting on whitespace.
        """
        line = line.lower()
        for pattern_re, replaced_str in _patterns_dict:
            line = pattern_re.sub(replaced_str, line)
        return line.split()

    return _basic_english_normalize

# Initialize tokenizer, stop words, and stemmer
tokenizer = get_tokenizer()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_essays(essays):
    """
    Preprocesses the list of essays using:
    - Tokenization
    - Stopword removal
    - Stemming
    - Rare word removal
    - Word to index mapping for building vocabulary
    :param essays: the list of essays to preprocess.
    :return: processed essays as lists of tokens and the vocab list.
    """
    processed_essays = []
    vocab = set()
    for essay in essays:
        # Tokenize the essay
        tokens = tokenizer(essay)
        # Remove all stopwords
        tokens = [token for token in tokens if token not in stop_words]
        # Stem remaining words
        tokens = [stemmer.stem(token) for token in tokens]
        # Remove rare words
        freq_dist = FreqDist(tokens)
        threshold = 2
        tokens = [token for token in tokens if freq_dist[token] > threshold]
        # Add tokens to list of processed essays
        processed_essays.append(tokens)
        # Add new words to vocab
        for token in tokens:
            vocab.add(token)
    return processed_essays, vocab

def map_vocab(processed_essays, vocab):
    vocab_size = len(vocab)
    essay_tensors = []
    # Create word to index mapping for the vocabulary
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    # Convert essay tokens to vocab indices
    for essay in processed_essays:
        essay_tensors.append(torch.LongTensor([word_to_idx[w] for w in essay]))
    return essay_tensors, vocab_size

# Create Dataset class for essays
class EssayDataset(Dataset):
    def __init__(self, text):
        self.text = text
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        return self.text[idx]

tokens, vocab = preprocess_essays(ai_human_df.iloc[:8, 0])
tensors, vocab_size = map_vocab(tokens, vocab)

print(vocab_size)
print(tensors[0])
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=50)

print(embedding(tensors[0]))