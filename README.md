# AI-/Human-Generated LSTM Text Classifier

---

## Introduction

The objective of this project is to design a neural network binary classifier that can identify instances of text created by generative AI and differentiate those instances from human-written text. It also provides experience with using the PyTorch library, the Long Short-Term Memory neural network architecture, and natural language processing techniques. 

---

## Necessary Files

The dataset used for training, validation, and testing can be found at the following link on Kaggle:
* https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

The secondary dataset used for generalization testing can be found at the following link on Science Direct:
* https://www.sciencedirect.com/science/article/pii/S2352340924002117

---

## Design

This section discusses the design choices made for the project, including preprocessing techniques and the chosen neural network architecture. Understanding the purposes of each of these choices is key for understanding and interpreting the performance of the model.

### Preprocessing

In order for the model to be successful in learning from the dataset, the input data must be preprocessed to isolate important features and reduce noise. First, we explore the dataset to learn about the structure of the raw data. The training data contains two columns: the first contains an essay, and the second contains a float that is either `1.0` if the text was written by an AI or `0.0` if it was not. For clarity, we convert this column's datatype to an integer. We also remove several problematic essay entries from the dataset by excluding any essays with a length shorter than 50 characters. Exploration reveals that these essays (of which there are only about 20 or so) mostly contained meaningless text generated by AIs that would not be useful for training.

Next, we visualize the class distribution of the dataset using matplotlib. The program saves this figure as `class-distribution.png` and shows that there are far more human-generated essays than AI-generated ones in the dataset. Understanding this imbalance is important for setting initial weights for Binary Cross-Entropy Loss later on.

![class-distribution](https://github.com/user-attachments/assets/50db36ba-a5bf-46bb-8efc-3e4ca7772632)

#### Essay Preprocessing Pipeline

The functions for essay preprocessing are mainly contained within `preprocessing.py` and `vocab.py`. The `generate_vocab` function within `vocab.py` allows generating a vocabulary for the model based on essays from the training dataset. The `VocabGenerator` object uses a tokenizer from `tokenizer.py` to convert essays into a list of tokens which, for the purposes of this model, correspond to individual words within an essay. Both a Set containing unique vocabulary words and a list containing all essay tokens are created. The list of all essay tokens allows for the creation of a frequency distribution for all words in the dataset. Using this, we can remove any rare or very common words from the vocabulary that are unlikely to contribute much value to the model. We also remove "stopwords" from the vocabulary, which are meaningless words such as "and" or "the" that do not contribute to the meaning of a sentence. Lastly, we stem the tokens, which reduces each word to its most basic form, removing things like specific conjugations. The remaining vocabulary words are stored in a dictionary mapping each word to a unique index.

The `EssayPreprocessor` in `preprocessing.py` similarly performs tokenization, stopword removal, and stemming, and then calls the `map_tokens_to_index` method from the `VocabGenerator` passed to it during initialization to convert essays into lists of indices. These indices allow for the model to create embeddings for each word, which attempt to capture the semantic meaning of each word. Each embedding is an n-dimensional vector of floating point values, where n is chosen during model initialization. The idea is that the vector obtained by subtracting the vectors for "man" and "woman" would return a similar vector as for subtracting "king" and "queen," thereby capturing the semantic difference between these pairs of words.

The model loads individual essays using PyTorch's `DataLoader`, which feeds the model input in batches. The construction of these batches is determined by the `collate_batch` function in `ai-text-classification.py`, which utilizes the essay processing pipeline to create pairs of processed essays and their corresponding labels (0 or 1). For each batch, the processed essays (which are currently lists of integer indices) are padded with zeroes such that each essay has a length equal to the longest essay in the batch. This is necessary for training the model, as all the inputs within a batch must be the same length. The original lengths of each sequence before padding are stored in a separate list. Lastly, the padded sequences, sequence lengths, and label lists are converted to tensors for the model. The padded sequences and label lists are sent to the GPU (as this is where the model will do its training), while the sequence lengths must be kept on the CPU. 

#### Alternative Pipeline

As an alternative option, the `EssayPreprocessor` class includes a pipeline that utilizes a subword tokenizer from HuggingFace. Using this tokenizer does not require initializing a `VocabGenerator` object. Subword tokenization can be more effective at capturing the meanings of longer or more rare words, since it considers the individual pieces of complex words rather than the words themselves. 

#### The LSTM Model

For this classification task, we use a Long Short-Term Memory neural network, which is a type of recurrent neural network (RNN) particularly useful for analyzing sequenced or time-series data as it allows previously seen information to influence its interpretation of current information.

The model contains a cell state, which represents the long-term memory, and a hidden state, which represents the short-term memory. The units of the LSTM model are comprised of a forget gate, an input gate, and an output gate. The forget gate determines the percentage of the current long-term memory to keep, using the current input and short-term memory with the weights and biases to do so. The input gate then uses the same inputs with different weights and biases to create a new long-term memory to sum with the last one. Lastly, the output gate creates a new short-term memory for the next LSTM unit (or the output if it is the last). LSTM models avoid the issue of exploding or vanishing gradients, which traditional RNNs experience, by avoiding the use of weights or biases in the calculation of long-term memories. Thus, it is not possible for repeated multiplication by a particular weight value to result in an extremely large or small value.

`essayLSTM.py` contains the definition for the model. It contains an embedding layer, which converts tensors of indices into embedding vectors. The embeddings are packed (removing the 0 indices from padding) and sent to the LSTM layer, the output of which is processed through a ReLU layer and finally a fully-connected layer to get the output.

---

## Training

Before starting model training, we initialize class weights corresponding to the imbalance of the class distribution noted during data exploration. This will ensure that misidentifications of the less prevalent class will yield higher loss values to make up for the lack of samples.

Then, we initialize the loss function and optimizer for the model. For this project, we utilize Cross Entropy Loss and the Adam optimizer. Adam allows for the learning rate of the model to be adjusted automatically during training, helping the model to reach optimal results.

### Training Epochs

During training, we run each batch of data through the model and propagate losses backwards to update weights and biases of the model. While LSTMs are good at avoiding exploding gradients, we still utilize gradient clipping to ensure that gradients remain within an acceptable range. Statistics about the accuracy and loss of the model are saved for plotting after training is complete.

### Validation

Between each epoch of training, the model is tested on a separate validation dataset (the split for which is 80:20). The accuracy, loss, precision, and recall of the model are recorded during validation. Precision quantifies how well the model avoids false positives (where human-written essays are classified as AI-generated), whereas recall quantifies how well the model avoids false negatives (where AI-generated essays are classified as human-written). We also calculate the F1 score for validation, which balances precision and recall scores.

### Cross-Validation

In order to guage whether the model is overfitting or not, it can be useful to compare the model's training metrics to cross-validation metrics. Cross-validation divides the training data into several "folds" and gives each fold a chance to be the validation data. This means that for five folds, the model will be fit five times. Taking the average score for these folds and comparing it to the score obtained from a standard single fit can reveal information about the performance of the model. If the cross-validation score is higher than the single fit score, the model may suffer from high variance (overfitting). On the other hand, if the scores are similar but poor, the model may suffer from high bias (underfitting).

### Plotting

Once training is complete, plots for training/validation loss and accuracy as well as validation precision, recall, and F1 score are constructed. These plots help with analyzing the performance of the model and can provide insight about how to tune model parameters for better performance.

One other plot is the receiver operating characteristic (ROC) curve, which plots the false positive rate versus the true positive rate for various "decision thresholds" for binary classification problems. The threshold refers to the value for which the output of the model must equal or exceed for the example to be classified as positive. For example, a threshold of `0.5` weights each binary class equally, whereas a threshold of `0.9` requires the model to be much more "confident" that an example is positive for it to be classified as such. Calculating the area under this curve helps to analyze model performance, since a higher area under the curve indicates a higher true positive rate for lower false positive rates. 

![loss-accuracy](https://github.com/user-attachments/assets/06123808-8da2-499e-9d82-ad4818979e9a)
![val-metrics](https://github.com/user-attachments/assets/c5d0028a-db80-4878-a88a-04e2a43b36d0)

---

## Testing

With the model complete, we can test its performance to evaluate how well it learned the important characteristics of AI-generated text that distinguish it from human-written text. The testing split is passed through the model, yielding an ROC score representing the overall performance of the model on unseen data. We also plot the ROC curve to visualize the true positive rate and false positive rate for different decision thresholds.

![image](https://github.com/user-attachments/assets/32c24d9f-c399-4a98-82e4-a54c248a7df8)

At the end of the main script, the model's state dictionary and parameters are saved to files for loading during testing. `model-test.py` loads these values and defines a function for predicting the class of new input using the model. The secondary testing dataset is passed through the model, yielding an accuracy value and ROC score for the new data. This testing accuracy acts as a final evaluation of the model's performance, which helps with determining the effectiveness of attempts to tune the model.

![image](https://github.com/user-attachments/assets/eb015f2b-6cc0-4b5a-8d61-50c88c274bb6)

---

## Feature Engineering/Tuning

Improving model performance requires examining the many parameters utilized in creating the model as well as the method of preprocessing the input data. There are many factors to consider:

* Vocabulary size
* Threshold for rare words
* Dataset input size for training
* Batch size
* Number of epochs
* Base learning rate
* Embedding size
* Model hidden size and number of stacked LSTM layers

We can also improve model performance using regularization techniques, such as L1 and L2 regularization. These techniques create additional loss for the model during each training iteration based on the weights of the model's parameters, helping with feature selection or helping to reduce the risk of overfitting to the training data.

Improving model performance by tuning each of these parameters is perhaps the most challenging part of creating a good model, but it is one of the most important steps because it directly influences the final testing accuracy of the model. So far, while the model is very accurate on the original dataset, it is only about 62% accurate on the secondary testing data, leaving much room for improvement.

---

## Future Steps

It is important to recognize that this classification task may be difficult for this neural network architecture, given the complexity of analyzing the characteristics of AI-generated text. Models that utilize transformer architecture may be better suited to perform this task. Learning the capabilities of this type of model and applying it to this dataset would be a good next step for improving performance.

There is also a possibility that the dataset used for training is not representative of the population of AI-generated responses. This would help to explain the difference in performance between the two datasets. Since the cross-validation scores show that the model does not suffer from extreme overfitting, this may very well be the case.
