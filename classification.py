#-----------------------------------------------------------------------------------------------------------------------
# AI Text Classification
#
# Python script for training a machine learning model for classifying AI-generated versus human-generated text.
#-----------------------------------------------------------------------------------------------------------------------

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, accuracy_score, recall_score, precision_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import time

# Create Dataset class for essays
class EssayDataset(Dataset):
    def __init__(self, dataframe):
        self.essays = dataframe["text"]
        self.labels = dataframe["generated"]
    def __len__(self):
        return len(self.essays)
    def __getitem__(self, idx):
        return self.essays.iloc[idx], self.labels.iloc[idx]

def create_collate_fn(pipeline):
    def collate_batch(batch):
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_list, label_list = [], []
        for _text, _label in batch:
            # Append label (no processing necessary)
            label_list.append(_label)
            # Process and append text
            processed_text = pipeline(_text)
            processed_text = torch.tensor(processed_text, dtype=torch.int64)
            text_list.append(processed_text)
        label_list = torch.tensor(label_list, dtype=torch.float)
        # Pad tensors so that batch elements are equal length
        padded_sequences = pad_sequence(text_list, batch_first=True, padding_value=0)
        sequence_lengths = torch.tensor([len(text) for text in text_list])
        # Send tensors to GPU
        return padded_sequences.to(model_device), sequence_lengths.to("cpu"), label_list.to(model_device) # lengths must be on CPU
    return collate_batch

def get_dataloaders(training_split, testing_split, batch_size, preprocessor):
    # Create training DataLoader
    train_dataloader = DataLoader(
        training_split,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=create_collate_fn(preprocessor),
        generator=torch.Generator().manual_seed(42)
    )

    # Create testing DataLoader
    test_dataloader = DataLoader(
        testing_split,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=create_collate_fn(preprocessor),
        generator=torch.Generator().manual_seed(42)
    )

    # Return DataLoaders
    return train_dataloader, test_dataloader

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

def get_loss_accuracy(metrics_dict: dict, threshold):
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

def get_recall_prec_f1(metrics_dict: dict, threshold):
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

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

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


def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs):
    sns.set_palette("Set1")
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Model Loss and Accuracy for Training and Validation")
    ax[1].set_ylim(0, 1)

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


def plot_prec_recall_f1(precisions, recalls, f1s, num_epochs):
    sns.set_palette("Set1")
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(6.4, 6.4))
    fig.suptitle("Model Validation Precision, Recall, and F1 Score")

    x_range = [x for x in range(1, num_epochs + 1)]
    x_ticks = [x for x in x_range if x % 2 == 0]

    fig.supxlabel("Epoch", fontsize=12)
    ax[0].set_ylim(0, 1)
    ax[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax[0].set_xticks(x_ticks)

    # Set up plot for Precision
    sns.lineplot(ax=ax[0], y=precisions, x=x_range)
    ax[0].set_title("Validation Precision")
    ax[0].set_ylabel("Precision")

    # Set up plot for Recall
    sns.lineplot(ax=ax[1], y=recalls, x=x_range)
    ax[1].set_title("Validation Recall")
    ax[1].set_ylabel("Recall")

    # Set up plot for F1 Score
    sns.lineplot(ax=ax[2], y=f1s, x=x_range)
    ax[2].set_title("Validation F1")
    ax[2].set_ylabel("F1 Score")

    # Increase spacing between plots and show
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()
    fig.savefig("val-metrics.png")
