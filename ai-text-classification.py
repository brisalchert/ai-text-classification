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
from torch.utils.data import Dataset, DataLoader, random_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import re
from tqdm import tqdm

# Load dataset from csv
ai_human_df = pd.read_csv('ai_human.csv')

# Fix class label data type
ai_human_df['generated'] = ai_human_df['generated'].astype(int)

# Preview data
# print(ai_human_df.head())
# print(ai_human_df.info())

# Visualize distribution of data
# sns.countplot(x=ai_human_df['generated'])
# plt.title('Class Distribution')
# plt.xlabel('Class')
# plt.ylabel('Count')
# plt.show()

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

def generate_tokens(essays):
    """
    Preprocesses the list of essays using:
    - Tokenization
    - Stopword removal
    - Stemming
    - Rare word removal
    :param essays: the list of essays to preprocess.
    :return: processed essays as lists of tokens and the vocab list.
    """
    processed_essays = []
    vocab = set()
    print("Tokenizing essays:")
    for essay in tqdm(essays):
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

def tokenize_and_generate_vocab(essays):
    tokens, vocab_list = generate_tokens(essays)
    vocab_to_idx = {word: i for i, word in enumerate(vocab_list)}
    return tokens, vocab_to_idx

def map_vocab(tokens_list, vocab_dict):
    essay_index_list = []
    # Convert essay tokens to vocab indices
    for tokens in tokens_list:
        essay_indices = [vocab_dict[w] for w in tokens]
        essay_index_list.append(essay_indices)
    return essay_index_list

def essay_processing_pipeline(essays):
    tokens, vocab_dict = tokenize_and_generate_vocab(essays)
    processed_essays = map_vocab(tokens, vocab_dict)
    return processed_essays, vocab_dict

# Create Dataset class for essays
class EssayDataset(Dataset):
    def __init__(self, dataframe):
        self.essays = dataframe['text']
        self.labels = dataframe['generated']
    def __len__(self):
        return len(self.essays)
    def __getitem__(self, idx):
        return self.essays.iloc[idx], self.labels.iloc[idx]

# Set device to CUDA GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    text_list, label_list, offsets = [], [], [0]
    for _text, _label in batch:
        # Append label (no processing necessary)
        label_list.append(_label)
        # Process and append text
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return text_list.to(device), label_list.to(device), offsets.to(device)

# Generate sample from data
sample_df = ai_human_df.sample(n=10000, random_state=42)

# Process essays and generate vocab
sample_df['text'], vocab = essay_processing_pipeline(sample_df['text'])
vocab_size = len(vocab)

# Create dataset for preprocessed data
essay_dataset = EssayDataset(sample_df)

split_train, split_valid = random_split(essay_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
train_dataloader = DataLoader(split_train, batch_size=16, shuffle=True, collate_fn=collate_batch)

train_features, train_labels, train_offsets = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
print(f"Offsets batch shape: {train_offsets.size()}")

for i in range(len(train_labels)):
    if i < 15:
        print("Features: " + str(train_features[train_offsets[i]:train_offsets[i + 1]]))
    else:
        print("Features: " + str(train_features[train_offsets[i]:]))

    print("Label: " + str(train_labels[i]))

# embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=50)
# Use nn.EmbeddingBag with all the tensors concatenated as a single tensor
# to avoid issues with non-matching tensor dimensions for batches