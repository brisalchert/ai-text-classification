#-----------------------------------------------------------------------------------------------------------------------
# AI Text Classification
#
# Python script for training a machine learning model for classifying AI-generated versus human-generated text.
#-----------------------------------------------------------------------------------------------------------------------

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import random_split, Subset
from classification import EssayDataset, get_dataloader, fit_evaluate, evaluate, plot_roc_curve, get_loss_accuracy, \
    get_recall_prec_f1, plot_loss_accuracy, plot_prec_recall_f1, get_roc_auc_score
from preprocessing import EssayPreprocessor
from essayLSTM import EssayLSTM
import pickle

# Load dataset from csv
ai_human_df = pd.read_csv("ai_human.csv")

# Set device to CUDA GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Generate sample from data
sample_df = ai_human_df.sample(n=51200, random_state=42)

# Create dataset object for iteration
essay_dataset = EssayDataset(sample_df)

# Set the batch size
batch_size = 64

# Split data
split_temp, split_test = random_split(essay_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
split_train, split_val = random_split(split_temp, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

# Create KFold object for cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize essay preprocessor
preprocessor = EssayPreprocessor()

# Create train, validation, and test DataLoaders
train_dataloader = get_dataloader(split_train, batch_size, preprocessor.huggingface_pipeline)
val_dataloader = get_dataloader(split_val, batch_size, preprocessor.huggingface_pipeline)
test_dataloader = get_dataloader(split_test, batch_size, preprocessor.huggingface_pipeline)

# Define number of epochs and initial learning rate
num_epochs = 15
learning_rate = 0.001

# Set model parameters
vocab_size = preprocessor.huggingface_tokenizer.vocab_size
embed_size = 200
hidden_size = 2
num_layers = 1

# Define decision threshold for classification
threshold = 0.5

# Define lambda_reg for L1 regularization
l1_lambda = 1e-7

total_accu = None

print("Starting training for cross-validation...")

# Fit model with cross-validation using pre-validation-split training set
cv_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(split_temp)):
    print(f"FOLD {fold + 1}:")

    train_loader_cv = get_dataloader(
        Subset(split_temp, train_idx),
        batch_size,
        preprocessor.huggingface_pipeline
    )

    val_loader_cv = get_dataloader(
        Subset(split_temp, val_idx),
        batch_size,
        preprocessor.huggingface_pipeline
    )

    # Initialize model for cross-validation
    cv_model = EssayLSTM(vocab_size, embed_size, hidden_size, num_layers, device)
    cv_model.to(device)

    # Initialize positive class weight for cross entropy loss [weight = (num_negative / num_positive)]
    num_positive = 0
    for data, _, labels in train_loader_cv:
        num_positive += sum(labels).item()
    pos_weight = torch.tensor((len(train_loader_cv) * batch_size - num_positive) / num_positive).to(device)

    # Initialize loss function and optimizer
    cv_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    cv_optimizer = torch.optim.Adam(cv_model.parameters(), lr=learning_rate)

    # Fit the model
    cv_results.append(fit_evaluate(
        cv_model, train_loader_cv, val_loader_cv, num_epochs, cv_optimizer,
        criterion=cv_criterion,
        l1_lambda=l1_lambda,
        logging=True
    ))

# Initialize the model for non-cross-validation
lstm_model = EssayLSTM(vocab_size, embed_size, hidden_size, num_layers, device)
lstm_model.to(device)

# Initialize positive class weight for cross entropy loss [weight = (num_negative / num_positive)]
num_positive = 0
for data, _, labels in train_dataloader:
    num_positive += sum(labels).item()
pos_weight = torch.tensor((len(train_dataloader) * batch_size - num_positive) / num_positive).to(device)

# Initialize loss function and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

print("Training on full train data...")

# Fit model on non-cross-validated training set
metrics = fit_evaluate(
    lstm_model, train_dataloader, val_dataloader, num_epochs, optimizer,
    criterion=criterion,
    l1_lambda=l1_lambda,
    logging=True
)

# Compare cross-validation to non-cross-validation using ROC AUC score
cv_roc_scores = []

for metric in cv_results:
    roc_scores = get_roc_auc_score(metric["val_metrics"])

    # Append the ROC AUC score from final epoch
    cv_roc_scores.append(roc_scores[-1])

cv_roc_score = sum(cv_roc_scores) / len(cv_roc_scores)
roc_score = get_roc_auc_score(metrics["val_metrics"])[-1]

print("\nCV average ROC AUC score: ", cv_roc_score)
print("Non-CV ROC AUC score: ", roc_score, "\n")

# Generate ROC curve for testing data
true, pred_proba, _ = evaluate(lstm_model, test_dataloader, criterion)
y_true = torch.cat(true).cpu()
y_pred_proba = torch.cat(pred_proba).cpu()

print("Testing data ROC AUC score", roc_auc_score(y_true, y_pred_proba), "\n")
plot_roc_curve(y_true, y_pred_proba)

# Calculate training metrics for plotting
train_accuracies, train_losses = get_loss_accuracy(metrics["train_metrics"], threshold)

# Calculate validation metrics for plotting
val_accuracies, val_losses = get_loss_accuracy(metrics["val_metrics"], threshold)
val_precisions, val_recalls, val_f1s = get_recall_prec_f1(metrics["val_metrics"], threshold)

# Plot accuracy and loss for training and validation
plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs)

# Plot precision, recall, and f1 for validation
plot_prec_recall_f1(val_precisions, val_recalls, val_f1s, num_epochs)

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
