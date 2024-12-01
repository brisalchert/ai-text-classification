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
from sklearn.metrics import roc_curve, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.utils.rnn import pad_sequence
from preprocessing import EssayPreprocessor
import time
from vocab import VocabGenerator
from essayLSTM import EssayLSTM
import pickle

# Load dataset from csv
ai_human_df = pd.read_csv("ai_human.csv")

# Fix class label data type
ai_human_df["generated"] = ai_human_df["generated"].astype(int)

# Remove short essays from dataset
ai_human_df = ai_human_df[~(ai_human_df["text"].str.len() <= 50)]

# Preview data
print(ai_human_df.head())
print(ai_human_df.info())

# Visualize distribution of data
sns.set_style("darkgrid")
sns.set_context("notebook")
ax = sns.countplot(x=ai_human_df["generated"])
ax.set_xticks([0, 1], ["Human-Generated", "AI-Generated"])
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
fig = ax.get_figure()
fig.savefig("class-distribution.png")

# Create Dataset class for essays
class EssayDataset(Dataset):
    def __init__(self, dataframe):
        self.essays = dataframe["text"]
        self.labels = dataframe["generated"]
    def __len__(self):
        return len(self.essays)
    def __getitem__(self, idx):
        return self.essays.iloc[idx], self.labels.iloc[idx]

# Set device to CUDA GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    return padded_sequences.to(device), sequence_lengths.to("cpu"), label_list.to(device) # lengths must be on CPU

def train(model, dataloader, optimizer, threshold=None, lambda_reg=None, epoch=None, loss_criterion=None, logging=False):
    """
    Trains a machine learning model with a given optimizer on the formatted essay data for one epoch. Can
    optionally output logging information to the console.
    :param model: The machine learning model to train.
    :param dataloader: The essay Dataloader with batches of preprocessed essays, essay lengths, and essay labels.
    :param optimizer: The optimizer to be used for training.
    :param threshold: The decision threshold for binary classification of output logits. 0.5 if None.
    :param lambda_reg: Alpha value for L1 regularization. 10 * alpha applied to L2 regularization. If None, no
    regularization is applied.
    :param epoch: The current epoch for training, if logging is enabled.
    :param loss_criterion: A custom loss criterion function, or BCEWithLogitsLoss if None.
    :param logging: If true, logs information to the console during training.
    :return: The true labels, output logits, and average loss for each batch.
    """
    # Define output variables
    y_true = []
    y_pred_proba = []
    losses = []

    # Set loss_criterion if necessary
    if loss_criterion is None:
        loss_criterion = nn.BCEWithLogitsLoss()

    # Set threshold if necessary
    if threshold is None:
        threshold = 0.5

    # Set model to training mode
    model.train()

    # Set log interval for output
    log_interval = len(dataloader) // 5
    start_time = time.time()

    # Prepare batch metric variables
    batch_count = 0
    batch_loss_sum = 0
    batch_accuracy_sum = 0

    # Loop through training data
    for index, (essays, essay_lengths, essay_labels) in enumerate(dataloader):
        # Prepare targets from labels
        targets = essay_labels.reshape(-1, 1)
        # Zero gradient from prior iterations
        optimizer.zero_grad()
        # Get model outputs and loss
        logits = model(essays, essay_lengths)
        loss = loss_criterion(logits, targets)
        if lambda_reg is not None:
            # Calculate L1 and L2 regularization penalties
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss += lambda_reg * l1_norm
            loss += lambda_reg * 10 * l2_norm
        # Create predicted labels as binary labels with shape matching labels from dataloader
        predicted_labels = (torch.sigmoid(logits).reshape(-1) >= threshold).float()
        # Apply backpropagation
        loss.backward()
        # Clip to avoid exploding gradient
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        # Step the optimizer forward
        optimizer.step()
        # Log training information
        if logging:
            # Increment batch metric variables
            batch_count += 1
            batch_loss_sum += loss.item()
            batch_accuracy_sum += accuracy_score(targets.tolist(), predicted_labels.tolist())

            if (index + 1) % log_interval == 0 and index > 0:
                # Calculate batch metrics
                batch_accuracy_avg = batch_accuracy_sum / batch_count
                batch_loss_avg = batch_loss_sum / batch_count

                # Calculate elapsed time
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| accuracy {:8.3f} "
                    "| loss {:8.5f}"
                    "| time {:8.3f}s |".format(
                        epoch, (index + 1), len(dataloader), batch_accuracy_avg, batch_loss_avg, elapsed
                    )
                )

                # Reset timer
                start_time = time.time()

                # Reset batch metrics
                batch_count = 0
                batch_loss_sum = 0
                batch_accuracy_sum = 0

        # Append model results to output variables
        y_true.append(targets)
        y_pred_proba.append(logits.detach())
        losses.append(loss.item())

    return y_true, y_pred_proba, losses

def evaluate(model, dataloader, loss_criterion=None, lambda_reg=None):
    """
    Evaluates a machine learning model on the given validation data.
    :param model: The machine learning model to evaluate.
    :param dataloader: The essay Dataloader with batches of preprocessed essays, essay lengths, and essay labels.
    :param loss_criterion: A custom loss criterion function, or BCEWithLogitsLoss if None.
    :param lambda_reg: Alpha value for L1 regularization. 10 * alpha applied to L2 regularization. If None, no
    regularization is applied.
    :return: The true labels, output logits, and average loss for each batch.
    """
    # Define output variables
    y_true = []
    y_pred_proba = []
    losses = []

    # Set loss_criterion if necessary
    if loss_criterion is None:
        loss_criterion = nn.BCEWithLogitsLoss()

    # Set model to evaluation mode
    model.eval()

    # Ensure no gradient modification during evaluation
    with torch.no_grad():
        for index, (essays, essay_lengths, essay_labels) in enumerate(dataloader):
            # Prepare target from label
            targets = essay_labels.reshape(-1, 1)
            # Get model outputs and loss
            logits = model(essays, essay_lengths)
            loss = loss_criterion(logits, targets)
            if lambda_reg is not None:
                # Calculate L1 and L2 regularization penalties
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                loss += lambda_reg * l1_norm
                loss += lambda_reg * 10 * l2_norm
            # Append model results to output variables
            y_true.append(targets)
            y_pred_proba.append(logits.detach())
            losses.append(loss.item())

    return y_true, y_pred_proba, losses

def fit_evaluate(model, train_loader, val_loader, epochs, optimizer, criterion=None,
                 threshold=None, l1_lambda=None, logging=False):
    """
    Trains and evaluates a machine learning model on the given essay data. Can optionally print logging information
    to the console during training and after each epoch.
    :param model: The machine learning model to train.
    :param train_loader: The preprocessed essay training data as a DataLoader.
    :param val_loader: The preprocessed essay validation data as a DataLoader.
    :param epochs: The number of epochs to train the model for.
    :param optimizer: The optimizer to use during training.
    :param criterion: The loss criterion to use. If None, uses BCEWithLogitsLoss.
    :param threshold: Decision threshold for binary classification. If None, uses 0.5.
    :param l1_lambda: Alpha value for L1 regularization. 10 * Alpha applied to L2 regularization. If None,
    no regularization is applied.
    :param logging: If true, outputs logging information to console during training and after each epoch.
    :return: A dictionary of two dictionaries with the outputs of training and validation for each epoch. Each
    dictionary contains three elements for each epoch: a nested list of true labels for each batch, a nested list
    of predicted labels for each batch, and a list of average losses for each batch.
    """
    # Initialize output variables
    train_metrics = {}
    validation_metrics = {}

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        # Train model
        train_metrics[epoch] = train(
            model, train_loader, optimizer,
            threshold=threshold,
            loss_criterion=criterion,
            lambda_reg=l1_lambda,
            epoch=epoch,
            logging=logging
        )
        # Evaluate the model
        validation_metrics[epoch] = evaluate(
            model, val_loader,
            loss_criterion=criterion
        )

        # Print epoch statistics with validation accuracy
        if logging:
            # Calculate validation accuracy and loss
            targets_list = validation_metrics[epoch][0]
            pred_list = validation_metrics[epoch][1]

            val_accuracy_sum = 0

            # Set threshold if necessary
            if threshold is None:
                threshold = 0.5

            for y_true, y_pred_proba in zip(targets_list, pred_list):
                y_true = y_true.cpu()
                y_pred = (torch.sigmoid(y_pred_proba).reshape(-1) >= threshold).float().cpu()
                val_accuracy_sum += accuracy_score(y_true, y_pred)

            val_accuracy_avg = val_accuracy_sum / len(targets_list)
            val_loss_avg = np.mean(validation_metrics[epoch][2])

            print("-" * 93)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | "
                "validation accuracy {:8.3f} | "
                "validation loss {:8.5f} |".format(
                    epoch, time.time() - epoch_start_time, val_accuracy_avg, val_loss_avg
                )
            )
            print("-" * 93)

    return {
        "train_metrics": train_metrics,
        "val_metrics": validation_metrics
    }

def get_loss_accuracy(metrics_dict: dict):
    accuracies = []
    losses = []

    for epoch in metrics_dict.keys():
        accuracy_sum = 0

        targets = metrics_dict[epoch][0]
        predicted = metrics_dict[epoch][1]

        for y_true, y_pred_proba in zip(targets, predicted):
            y_true = y_true.cpu()
            y_pred = (torch.sigmoid(y_pred_proba).reshape(-1) >= threshold).float().cpu()

            accuracy_sum += accuracy_score(y_true, y_pred)

        loss_values = metrics_dict[epoch][2]

        # Epoch accuracy is sum of batch accuracies divided by number of batches
        epoch_accuracy = accuracy_sum / len(metrics_dict[epoch][0])
        epoch_loss = np.mean(loss_values)

        # Append epoch accuracy and loss to the lists
        accuracies.append(epoch_accuracy)
        losses.append(epoch_loss)

    # Return lists
    return accuracies, losses

def get_recall_prec_f1(metrics_dict: dict):
    recalls = []
    precisions = []
    f1s = []

    for epoch in metrics_dict.keys():
        recall_sum = 0
        precision_sum = 0
        f1_sum = 0

        targets = metrics_dict[epoch][0]
        predicted = metrics_dict[epoch][1]

        for y_true, y_pred_proba in zip(targets, predicted):
            y_true = y_true.cpu()
            y_pred = (torch.sigmoid(y_pred_proba).reshape(-1) >= threshold).float().cpu()

            recall_sum += recall_score(y_true, y_pred)
            precision_sum += precision_score(y_true, y_pred)
            f1_sum += f1_score(y_true, y_pred)

        batch_count = len(metrics_dict[epoch][0])

        # Calculate epoch validation metrics
        epoch_recall = recall_sum / batch_count
        epoch_precision = precision_sum / batch_count
        epoch_f1 = f1_sum / batch_count

        # Append epoch validation metrics to lists
        recalls.append(epoch_recall)
        precisions.append(epoch_precision)
        f1s.append(epoch_f1)

    # Return lists
    return recalls, precisions, f1s

# Generate sample from data
sample_df = ai_human_df.sample(n=40000, random_state=42)

# Create dataset object for iteration
essay_dataset = EssayDataset(sample_df)

# Set the batch size
batch_size = 64

# Split data
split_train, split_test = random_split(essay_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

# Create KFold object
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

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
num_epochs = 5
learning_rate = 0.001

# Set model parameters
vocab_size = vocab.get_vocab_size()
embed_size = 25
hidden_size = 2
num_layers = 1

# Define decision threshold for classification
threshold = 0.5

# Define lambda_reg for L1 regularization
l1_lambda = 1e-7

total_accu = None

print("Starting training for cross-validation...")

# Fit model on cross-validation set
for fold, (train_idx, val_idx) in enumerate(kfold.split(split_train)):
    print(f"FOLD {fold + 1}:")

    train_loader_cv = DataLoader(
        Subset(essay_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        generator = torch.Generator().manual_seed(42)
    )
    val_loader_cv = DataLoader(
        Subset(essay_dataset, val_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        generator=torch.Generator().manual_seed(42)
    )

    # Initialize model for cross-validation
    cv_model = EssayLSTM(vocab_size, embed_size, hidden_size, num_layers, device)
    cv_model.to(device)

    # Initialize weights for cross entropy loss [weight = (total / (num_per_class * num_classes))]
    pos_weight = torch.tensor(305797 / 181438).to(device)

    # Initialize loss function and optimizer
    cv_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    cv_optimizer = torch.optim.Adam(cv_model.parameters(), lr=learning_rate)

    # Fit the model
    fit_evaluate(
        cv_model, train_loader_cv, val_loader_cv, num_epochs, cv_optimizer,
        criterion=cv_criterion,
        l1_lambda=l1_lambda,
        logging=True
    )

# Initialize the model for non-cross-validation
lstm_model = EssayLSTM(vocab_size, embed_size, hidden_size, num_layers, device)
lstm_model.to(device)

# Initialize weights for cross entropy loss [weight = (total / (num_per_class * num_classes))]
pos_weight = torch.tensor(305797 / 181438).to(device)

# Initialize loss function and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

print("Training on full train data...")

# Fit model on non-cross-validated training set
metrics = fit_evaluate(
    lstm_model, train_dataloader, test_dataloader, num_epochs, optimizer,
    criterion=criterion,
    l1_lambda=l1_lambda,
    logging=True
)

# Generate ROC Curve for validation data
with torch.no_grad():
    true_labels = []
    predictions = []
    for idx, (text, lengths, label) in enumerate(test_dataloader):
        logits = lstm_model(text, lengths).cpu()
        y_score = torch.sigmoid(logits).squeeze(-1).tolist()
        predictions.extend(y_score)
        true_labels.extend(label.cpu().tolist())

    np.array(true_labels)
    np.array(predictions)
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    roc_df["cutoff"] = tpr - fpr

    optimal_index = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_index]
    print("Optimal threshold: ", optimal_threshold)

    print(roc_df.sort_values("cutoff", ascending=False))

    plt.plot(fpr, tpr, marker=".")
    plt.title("ROC Curve for Validation Data")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

# Calculate training metrics for plotting
train_accuracies, train_losses = get_loss_accuracy(metrics["train_metrics"])

# Calculate validation metrics for plotting
val_accuracies, val_losses = get_loss_accuracy(metrics["val_metrics"])
val_precisions, val_recalls, val_f1s = get_recall_prec_f1(metrics["val_metrics"])

# Plot accuracy and loss for training and validation
sns.set_palette("Set1")
fig, ax = plt.subplots(2, 1, sharex=True)
fig.suptitle("Model Loss and Accuracy for Training and Validation")
ax[1].set_ylim(0,1)

x_range = [x for x in range(1, num_epochs + 1)]
x_ticks = [x for x in x_range if x % 2 == 0]

ax[0].set_xticks(x_ticks)

# Prepare loss and accuracy data for multiline plot
loss_df = pd.DataFrame({
    "Epoch": x_range,
    "Training Loss": train_losses,
    "Validation Loss": val_losses,
})

acc_df = pd.DataFrame({
    "Epoch": x_range,
    "Training Accuracy": train_accuracies,
    "Validation Accuracy": val_accuracies,
})

# Convert DataFrames from wide to long format (one column for all measurements)
loss_df = pd.melt(loss_df, id_vars=["Epoch"])
acc_df = pd.melt(acc_df, id_vars=["Epoch"])
loss_df.rename(columns={"value": "Loss"}, inplace=True)
acc_df.rename(columns={"value": "Accuracy"}, inplace=True)

# Set up plot for Loss
sns.lineplot(ax=ax[0], data=loss_df, y="Loss", x="Epoch", hue="variable")
ax[0].set_title("Training and Validation Loss")
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(handles=handles, labels=labels)

# Set up plot for Accuracy
sns.lineplot(ax=ax[1], data=acc_df, y="Accuracy", x="Epoch", hue="variable")
ax[1].set_title("Training and Validation Accuracy")
handles, labels = ax[1].get_legend_handles_labels()
ax[1].legend(handles=handles, labels=labels)

# Increase spacing between plots and show
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()
fig.savefig("loss-accuracy.png")

# Plot precision, recall, and f1 for validation
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6.4, 6.4))
fig.suptitle("Model Validation Precision, Recall, and F1 Score")
fig.supxlabel("Epoch", fontsize=12)
ax[0].set_ylim(0,1)
ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax[0].set_xticks(x_ticks)

# Set up plot for Precision
sns.lineplot(ax=ax[0], y=val_precisions, x=x_range)
ax[0].set_title("Validation Precision")
ax[0].set_ylabel("Precision")

# Set up plot for Recall
sns.lineplot(ax=ax[1], y=val_recalls, x=x_range)
ax[1].set_title("Validation Recall")
ax[1].set_ylabel("Recall")

# Set up plot for F1 Score
sns.lineplot(ax=ax[2], y=val_f1s, x=x_range)
ax[2].set_title("Validation F1")
ax[2].set_ylabel("F1 Score")

# Increase spacing between plots and show
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()
fig.savefig("val-metrics.png")

# Save vocabulary
with open("vocab.pkl", "wb") as f:
    # noinspection PyTypeChecker
    pickle.dump(vocab.get_vocab_dictionary(), f)

# Save the model's state dictionary
torch.save(lstm_model.state_dict(), "ai-text-model.pt")

# Save model parameters
model_params = {
    "vocab_size": vocab_size,
    "embed_size": embed_size,
    "hidden_size": hidden_size,
    "num_layers": num_layers
}

with open("model-params.pkl", "wb") as f:
    # noinspection PyTypeChecker
    pickle.dump(model_params, f)
