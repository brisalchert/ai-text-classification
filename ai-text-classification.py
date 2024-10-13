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
import time
from tokenizer import get_tokenizer
from vocab import VocabGenerator
from essayLSTM import EssayLSTM
import pickle

# Load dataset from csv
ai_human_df = pd.read_csv('ai_human.csv')

# Fix class label data type
ai_human_df['generated'] = ai_human_df['generated'].astype(int)

# Preview data
print(ai_human_df.head())
print(ai_human_df.info())

# Visualize distribution of data
sns.set_style('darkgrid')
sns.set_context('notebook')
ax = sns.countplot(x=ai_human_df['generated'])
ax.set_xticks([0, 1], ['Human-Generated', 'AI-Generated'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
fig = ax.get_figure()
fig.savefig('class-distribution.png')

# Initialize tokenizer, stop words, and stemmer
tokenizer = get_tokenizer()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def generate_tokens(essay):
    # Tokenize the essay
    tokens = tokenizer(essay)
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # Stem remaining words
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def essay_processing_pipeline(essay):
    # Generate tokens
    tokens = generate_tokens(essay)
    # Map tokens to indices for embedding
    indices = vocab.map_tokens_to_index(tokens)
    return indices

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
        processed_text = essay_processing_pipeline(_text)
        processed_text = torch.tensor(processed_text, dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    # Pad tensors so that batch elements are equal length
    padded_sequences = pad_sequence(text_list, batch_first=True, padding_value=0)
    sequence_lengths = torch.tensor([len(text) for text in text_list])
    # Send tensors to GPU
    return padded_sequences.to(device), sequence_lengths.to('cpu'), label_list.to(device) # lengths must be on CPU

def train(dataloader, loss_criterion, optimizer):
    model.train()
    total_acc, total_count = 0, 0
    running_train_loss, running_train_acc = 0, 0
    log_interval = 200
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
        if (idx + 1) % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                '| epoch {:3d} | {:5d}/{:5d} batches '
                '| accuracy {:8.3f} '
                '| time {:8.3f}s |'.format(
                    epoch, (idx + 1), len(dataloader), total_acc / total_count, elapsed
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
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for idx, (text, lengths, label) in enumerate(dataloader):
            predicted_label = model(text, lengths)
            loss = loss_criterion(predicted_label, label)
            running_val_loss += loss.item()
            running_val_acc += (predicted_label.argmax(1) == label).sum().item()
            # Add true and predicted labels to lists
            true_labels.append(label)
            predicted_labels.append(predicted_label.argmax(1))

    # Convert true and predicted label lists to tensors
    true_labels = torch.cat(true_labels, dim=0)
    predicted_labels = torch.cat(predicted_labels, dim=0)

    # Calculate confusion matrix values for precision, recall, and F1 score
    true_positives = ((true_labels == 1) & (predicted_labels == 1)).sum().item()
    false_positives = ((true_labels == 0) & (predicted_labels == 1)).sum().item()
    false_negatives = ((true_labels == 1) & (predicted_labels == 0)).sum().item()

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    avg_val_loss = running_val_loss / (len(dataloader) * dataloader.batch_size)
    avg_val_acc = running_val_acc / (len(dataloader) * dataloader.batch_size)
    return avg_val_loss, avg_val_acc, precision, recall, f1

# Generate sample from data
sample_df = ai_human_df.sample(n=20000, random_state=42)

# Create dataset object for iteration
essay_dataset = EssayDataset(sample_df)

# Set the batch size
batch_size = 16

# Split data
split_train, split_test = random_split(essay_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

# Generate vocab using training data
vocab = VocabGenerator(split_train[:][0])

# Create train and test DataLoaders
train_dataloader = DataLoader(split_train, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(split_test, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

# Define number of epochs and initial learning rate
num_epochs = 20
learning_rate = 0.001

# Set model parameters
num_class = len(set([label for (text, label) in split_train]))
vocab_size = vocab.get_vocab_size()
embed_size = 64
hidden_size = 2
num_layers = 1

# Initialize the model
model = EssayLSTM(vocab_size, embed_size, hidden_size, num_layers, num_class, device)
model.to(device)

# Initialize weights for cross entropy loss [weight = (total / (num_per_class * num_classes))]
weights = torch.tensor([(487235 / (305797 * 2)), (487235 / (181438 * 2))]).to(device)

# Initialize loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_accu = None

train_losses = []
train_accs = []
val_losses = []
val_accs = []
val_precisions = []
val_recalls = []
val_f1s = []

print("Starting Training...")

for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    # Train model
    current_train_loss, current_train_acc = train(train_dataloader, criterion, optimizer)
    current_val_loss, current_val_acc, precision, recall, f1 = evaluate(test_dataloader, criterion)

    train_losses.append(current_train_loss)
    train_accs.append(current_train_acc)
    val_losses.append(current_val_loss)
    val_accs.append(current_val_acc)
    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1s.append(f1)

    # Print epoch statistics with validation accuracy
    print('-' * 119)
    print(
        '| end of epoch {:3d} | time: {:5.2f}s | '
        'validation accuracy {:8.3f} | '
        'precision {:8.3f} | '
        'recall {:8.3f} | '
        'f1 {:8.3f} |'.format(
            epoch, time.time() - epoch_start_time, current_val_acc, precision, recall, f1
        )
    )
    print('-' * 119)

# Plot accuracy and loss for training and validation
sns.set_palette('Set1')
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
fig.suptitle('Model Loss and Accuracy for Training and Validation')
ax[0].set_ylim(0,1)

x_range = [x for x in range(1, num_epochs + 1)]
x_ticks = [x for x in x_range if x % 2 == 0]

ax[0].set_xticks(x_ticks)

# Prepare loss and accuracy data for multiline plot
loss_df = pd.DataFrame({
    'Epoch': x_range,
    'Training Loss': train_losses,
    'Validation Loss': val_losses
})

acc_df = pd.DataFrame({
    'Epoch': x_range,
    'Training Accuracy': train_accs,
    'Validation Accuracy': val_accs
})

# Convert DataFrames from wide to long format (one column for all measurements)
loss_df = pd.melt(loss_df, id_vars=['Epoch'])
acc_df = pd.melt(acc_df, id_vars=['Epoch'])
loss_df.rename(columns={'value': 'Loss'}, inplace=True)
acc_df.rename(columns={'value': 'Accuracy'}, inplace=True)

# Set up plot for Loss
sns.lineplot(ax=ax[0], data=loss_df, y='Loss', x='Epoch', hue='variable')
ax[0].set_title('Training and Validation Loss')
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles=handles, labels=labels)

# Set up plot for Accuracy
sns.lineplot(ax=ax[1], data=acc_df, y='Accuracy', x='Epoch', hue='variable')
ax[1].set_title('Training and Validation Accuracy')
handles, labels = ax[1].get_legend_handles_labels()
ax[1].legend(handles=handles, labels=labels)

# Increase spacing between plots and show
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()
fig.savefig('loss-accuracy.png')

# Plot precision, recall, and f1 for validation
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6.4, 6.4))
fig.suptitle('Model Validation Precision, Recall, and F1 Score')
fig.supxlabel('Epoch', fontsize=12)
ax[0].set_ylim(0,1)
ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax[0].set_xticks(x_ticks)

# Set up plot for Precision
sns.lineplot(ax=ax[0], y=val_precisions, x=x_range)
ax[0].set_title('Validation Precision')
ax[0].set_ylabel('Precision')

# Set up plot for Recall
sns.lineplot(ax=ax[1], y=val_recalls, x=x_range)
ax[1].set_title('Validation Recall')
ax[1].set_ylabel('Recall')

# Set up plot for F1 Score
sns.lineplot(ax=ax[2], y=val_f1s, x=x_range)
ax[2].set_title('Validation F1')
ax[2].set_ylabel('F1 Score')

# Increase spacing between plots and show
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()
fig.savefig('val-metrics.png')

# Save the model's state dictionary
torch.save(model.state_dict(), 'ai-text-model.pt')

# Save model parameters
model_params = {
    'vocab_size': vocab_size,
    'embed_size': embed_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'num_class': num_class
}

with open('model-params.pkl', 'wb') as f:
    # noinspection PyTypeChecker
    pickle.dump(model_params, f)
