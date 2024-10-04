#-----------------------------------------------------------------------------------------------------------------------
# AI Text Classification
#
# Python script for training a CNN for classifying AI-generated versus human-generated text.
#-----------------------------------------------------------------------------------------------------------------------

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import re
from tqdm import tqdm
import time

# Load dataset from csv
ai_human_df = pd.read_csv('ai_human.csv')

# Fix class label data type
ai_human_df['generated'] = ai_human_df['generated'].astype(int)

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_batch(batch):
    text_list, label_list = [], []
    for _text, _label in batch:
        # Append label (no processing necessary)
        label_list.append(_label)
        # Process and append text
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    # Pad tensors so that batch elements are equal length
    padded_sequences = pad_sequence(text_list, batch_first=True, padding_value=0)
    sequence_lengths = torch.tensor([len(text) for text in text_list])
    # Send tensors to GPU
    return padded_sequences.to(device), sequence_lengths.to('cpu'), label_list.to(device) # lengths must be on CPU

class EssayLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes):
        super(EssayLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, sequences, lengths):
        # Perform embedding
        embedded = self.embedding(sequences)
        h0 = torch.zeros(self.num_layers, embedded.size(0), self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, embedded.size(0), self.hidden_size, device=device)
        # Pack the embedded sequences
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        # Propagate embeddings through LSTM layer
        out, (hn, cn) = self.lstm(packed, (h0, c0))
        hn = hn.view(-1, self.hidden_size) # Reshape for following Dense layer
        out = self.relu(hn)
        out = self.fc(out)
        return out

def train(dataloader, loss_criterion, optimizer):
    model.train()
    total_acc, total_count = 0, 0
    running_train_loss, running_train_acc = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (text, lengths, label) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, lengths)
        loss = loss_criterion(predicted_label, label)
        running_train_loss += loss.item()
        loss.backward()
        # Clip to avoid exploding gradient
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        running_train_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                '| epoch {:3d} | {:5d}/{:5d} batches '
                '| accuracy {:8.3f} '
                '| time {:8.3f}s |'.format(
                    epoch, idx, len(dataloader), total_acc / total_count, elapsed
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()

    avg_train_loss = running_train_loss / (len(dataloader) * dataloader.batch_size)
    avg_train_acc = running_train_acc / (len(dataloader) * dataloader.batch_size)
    return avg_train_loss, avg_train_acc

def evaluate(dataloader, loss_criterion):
    model.eval()
    running_val_loss, running_val_acc = 0, 0

    with torch.no_grad():
        for idx, (text, lengths, label) in enumerate(dataloader):
            predicted_label = model(text, lengths)
            loss = loss_criterion(predicted_label, label)
            running_val_loss += loss.item()
            running_val_acc += (predicted_label.argmax(1) == label).sum().item()
    avg_val_loss = running_val_loss / (len(dataloader) * dataloader.batch_size)
    avg_val_acc = running_val_acc / (len(dataloader) * dataloader.batch_size)
    return avg_val_loss, avg_val_acc

# Generate sample from data
sample_df = ai_human_df.sample(n=10000, random_state=42)

# Process essays and generate vocab
sample_df['text'], vocab = essay_processing_pipeline(sample_df['text'])

# Create dataset for preprocessed data
essay_dataset = EssayDataset(sample_df)

# Set the batch size
batch_size = 4

# Split data and create DataLoaders
split_train, split_test = random_split(essay_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
train_dataloader = DataLoader(split_train, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(split_test, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

# Define number of epochs and initial learning rate
num_epochs = 10
learning_rate = 0.1

# Set model parameters
num_class = len(set([label for (text, label) in split_train]))
vocab_size = len(vocab)
embed_size = 64
hidden_size = 2
num_layers = 1

# Initialize the model
model = EssayLSTM(vocab_size, embed_size, hidden_size, num_layers, num_class)
model.to(device)

# Initialize weights for cross entropy loss [weight = (total / (num_per_class * num_classes))]
weights = torch.tensor([(487235 / (305797 * 2)), (487235 / (181438 * 2))]).to(device)

# Initialize loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

total_accu = None

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    # Train model
    current_train_loss, current_train_acc = train(train_dataloader, criterion, optimizer)
    current_val_loss, current_val_acc = evaluate(test_dataloader, criterion)

    train_losses.append(current_train_loss)
    train_accs.append(current_train_acc)
    val_losses.append(current_val_loss)
    val_accs.append(current_val_acc)

    # Step the learning rate scheduler for every epoch the accuracy decreases
    if total_accu is not None and total_accu > current_val_acc:
        scheduler.step()
    else:
        total_accu = current_val_acc

    # Print epoch statistics with validation accuracy
    print('-' * 72)
    print(
        '| end of epoch {:3d} | time: {:5.2f}s | '
        'validation accuracy {:8.3f}       |'.format(
            epoch, time.time() - epoch_start_time, current_val_acc
        )
    )
    print('-' * 72)

# Plot accuracy and loss for training and validation
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.suptitle('Model statistics')
ax[0, 0].set_ylim(0,1)

sns.lineplot(ax=ax[0, 0], data=train_losses)
ax[0, 0].set_title('Training Loss')

sns.lineplot(ax=ax[0, 1], data=train_accs)
ax[0, 1].set_title('Training Accuracy')

sns.lineplot(ax=ax[1, 0], data=val_losses)
ax[1, 0].set_title('Validation Loss')

sns.lineplot(ax=ax[1, 1], data=val_accs)
ax[1, 1].set_title('Validation Accuracy')

plt.show()

# gen_essay_label = {0: 'human-generated', 1: 'AI-generated'}
#
# def predict(essay, essay_pipeline):
#     with torch.no_grad():
#         text, _ = essay_pipeline(essay)
#         text = torch.tensor(text)
#         output = model(text)
#         return output.argmax(1).item()
#
# ex_text_str = ai_human_df.sample(n=1)
# print('Actual label: ' + gen_essay_label[ex_text_str.iloc[0, 1]])
#
# model = model.to('cpu')
#
# print('This text is %s.' % gen_essay_label[predict(ex_text_str, essay_processing_pipeline)])
