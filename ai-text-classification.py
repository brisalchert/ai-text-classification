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
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from preprocessing import EssayPreprocessor
import time
from vocab import VocabGenerator
from essayLSTM import EssayLSTM
import pickle

# Load dataset from csv
ai_human_df = pd.read_csv('ai_human.csv')

# Fix class label data type
ai_human_df['generated'] = ai_human_df['generated'].astype(int)

# Remove short essays from dataset
ai_human_df = ai_human_df[~(ai_human_df['text'].str.len() <= 50)]

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
        processed_text = preprocessor.essay_processing_pipeline(_text)
        processed_text = torch.tensor(processed_text, dtype=torch.int64)
        text_list.append(processed_text)
    label_list = torch.tensor(label_list, dtype=torch.float)
    # Pad tensors so that batch elements are equal length
    padded_sequences = pad_sequence(text_list, batch_first=True, padding_value=0)
    sequence_lengths = torch.tensor([len(text) for text in text_list])
    # Send tensors to GPU
    return padded_sequences.to(device), sequence_lengths.to('cpu'), label_list.to(device) # lengths must be on CPU

def train(dataloader, loss_criterion, optimizer, threshold, lambda_reg):
    model.train()
    total_acc, total_loss, total_count = 0, 0, 0
    running_train_loss, running_train_acc = 0, 0
    log_interval = len(dataloader) / 5
    start_time = time.time()

    for idx, (text, lengths, label) in enumerate(dataloader):
        #Prepare target from label
        target = label.reshape(-1, 1)
        optimizer.zero_grad()
        logit = model(text, lengths)
        loss = loss_criterion(logit, target)
        # Calculate L1 and L2 regularization penalties
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        l2_norm = sum(p.pow(2).sum() for p in model.parameters())
        loss += lambda_reg * l1_norm
        loss += lambda_reg * 10 * l2_norm
        # Create predicted label as binary label with shape matching label from dataloader
        predicted_label = (torch.sigmoid(logit).reshape(-1) >= threshold).float()
        total_loss += loss.item()
        running_train_loss += loss.item()
        loss.backward()
        # Clip to avoid exploding gradient
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_acc += (predicted_label == label).sum().item()
        running_train_acc += (predicted_label == label).sum().item()
        total_count += label.size(0)
        if (idx + 1) % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                '| epoch {:3d} | {:5d}/{:5d} batches '
                '| accuracy {:8.3f} '
                '| loss {:8.5f}'
                '| time {:8.3f}s |'.format(
                    epoch, (idx + 1), len(dataloader), total_acc / total_count, total_loss / total_count, elapsed
                )
            )
            total_acc, total_loss, total_count = 0, 0, 0
            start_time = time.time()

    avg_train_loss = running_train_loss / (len(dataloader) * dataloader.batch_size)
    avg_train_acc = running_train_acc / (len(dataloader) * dataloader.batch_size)
    return avg_train_loss, avg_train_acc

def evaluate(dataloader, loss_criterion, threshold, lambda_reg):
    model.eval()
    running_val_loss, running_val_acc = 0, 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for idx, (text, lengths, label) in enumerate(dataloader):
            # Prepare target from label
            target = label.reshape(-1, 1)
            logit = model(text, lengths)
            loss = loss_criterion(logit, target)
            # Calculate L1 and L2 regularization penalties
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss += lambda_reg * l1_norm
            loss += lambda_reg * 10 * l2_norm
            # Create predicted label as binary label with shape matching label from dataloader
            predicted_label = (torch.sigmoid(logit).reshape(-1) >= threshold).float()
            running_val_loss += loss.item()
            running_val_acc += (predicted_label == label).sum().item()
            # Add true and predicted labels to lists
            true_labels.append(label)
            predicted_labels.append(predicted_label)

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
sample_df = ai_human_df.sample(n=51200, random_state=42)

# Create dataset object for iteration
essay_dataset = EssayDataset(sample_df)

# Set the batch size
batch_size = 64

# Split data
split_train, split_test = random_split(essay_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

# Generate vocab using training data
vocab = VocabGenerator(essays=split_train[:][0])

# Initialize essay preprocessor
preprocessor = EssayPreprocessor(vocab)

# Create train and test DataLoaders
train_dataloader = DataLoader(
    split_train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch,
    generator=torch.Generator().manual_seed(42)
)
test_dataloader = DataLoader(
    split_test,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch,
    generator=torch.Generator().manual_seed(42)
)

# Define number of epochs and initial learning rate
num_epochs = 20
learning_rate = 0.001

# Set model parameters
vocab_size = vocab.get_vocab_size()
embed_size = 200
hidden_size = 4
num_layers = 2

# Define decision threshold for classification
threshold = 0.7

# Define lambda_reg for L1 regularization
l1_lambda = 1e-7

# Initialize the model
model = EssayLSTM(vocab_size, embed_size, hidden_size, num_layers, device)
model.to(device)

# Initialize weights for cross entropy loss [weight = (total / (num_per_class * num_classes))]
pos_weight = torch.tensor(305797 / 181438).to(device)

# Initialize loss function and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
    current_train_loss, current_train_acc = train(train_dataloader, criterion, optimizer,
                                                  threshold, l1_lambda)
    current_val_loss, current_val_acc, precision, recall, f1 = evaluate(test_dataloader, criterion,
                                                                        threshold, l1_lambda)

    train_losses.append(current_train_loss)
    train_accs.append(current_train_acc)
    val_losses.append(current_val_loss)
    val_accs.append(current_val_acc)
    val_precisions.append(precision)
    val_recalls.append(recall)
    val_f1s.append(f1)

    # Print epoch statistics with validation accuracy
    print('-' * 147)
    print(
        '| end of epoch {:3d} | time: {:5.2f}s | '
        'validation accuracy {:8.3f} | '
        'validation loss {:8.5f} | '
        'precision {:8.3f} | '
        'recall {:8.3f} | '
        'f1 {:8.3f} |'.format(
            epoch, time.time() - epoch_start_time, current_val_acc, current_val_loss, precision, recall, f1
        )
    )
    print('-' * 147)

# Generate ROC Curve for validation data
with torch.no_grad():
    true_labels = []
    predictions = []
    for idx, (text, lengths, label) in enumerate(test_dataloader):
        logits = model(text, lengths).cpu()
        y_score = torch.sigmoid(logits).squeeze(-1).tolist()
        predictions.extend(y_score)
        true_labels.extend(label.cpu().tolist())

    np.array(true_labels)
    np.array(predictions)
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    roc_df['cutoff'] = tpr - fpr

    optimal_index = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_index]
    print("Optimal threshold: ", optimal_threshold)

    print(roc_df.sort_values("cutoff", ascending=False))

    plt.plot(fpr, tpr, marker='.')
    plt.title("ROC Curve for Validation Data")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

# Plot accuracy and loss for training and validation
sns.set_palette('Set1')
fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle('Model Loss and Accuracy for Training and Validation')
ax[1].set_ylim(0,1)

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

# Save vocabulary
with open('vocab.pkl', 'wb') as f:
    # noinspection PyTypeChecker
    pickle.dump(vocab.get_vocab_dictionary(), f)

# Save the model's state dictionary
torch.save(model.state_dict(), 'ai-text-model.pt')

# Save model parameters
model_params = {
    'vocab_size': vocab_size,
    'embed_size': embed_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers
}

with open('model-params.pkl', 'wb') as f:
    # noinspection PyTypeChecker
    pickle.dump(model_params, f)
