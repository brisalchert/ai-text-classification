import pickle
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import *
from essayLSTM import EssayLSTM
from preprocessing import EssayPreprocessor
from vocab import VocabGenerator

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Text Classification")
        self.setStyleSheet("background-color: rgb(120, 120, 120);")

        title_label = QLabel("AI Text Classifier")
        title_label.setFont(QFont("Bahnschrift", 40))
        title_label.setAlignment(Qt.AlignHCenter)
        title_label.setStyleSheet("color: white;")

        self.input_textbox = QTextEdit()
        self.input_textbox.setFont(QFont("Cascadia Code", 10))
        self.input_textbox.setText("")
        self.input_textbox.setFixedSize(400, 300)
        self.input_textbox.setStyleSheet("""
            background-color: rgb(240, 240, 240);
            border: 2px solid black;
            color: black;
        """)

        textbox_layout = QHBoxLayout()
        textbox_layout.addWidget(self.input_textbox)

        self.predict_button = QPushButton("Predict")
        self.predict_button.setFixedWidth(200)
        self.predict_button.setFont(QFont("Cascadia Code", 10))
        self.predict_button.setStyleSheet("""
            background-color: rgb(60, 60, 60);
            color: white;
        """)
        self.predict_button.clicked.connect(self.get_prediction)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.predict_button)

        output_label = QLabel("Output: ")
        output_label.setFont(QFont("Bahnschrift", 18))
        output_label.setStyleSheet("color: white;")
        self.output = QLineEdit()
        self.output.setEnabled(False)
        self.output.setFont(QFont("Cascadia Code", 10))
        self.output.setPlaceholderText("Model output")
        self.output.setFixedWidth(300)
        self.output.setStyleSheet("""
            background-color: rgb(240, 240, 240);
            color: black;
            border: 2px solid black;
        """)

        output_layout = QHBoxLayout()
        output_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output)
        output_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addLayout(textbox_layout)
        layout.addLayout(button_layout)
        layout.addLayout(output_layout)
        layout.setAlignment(Qt.AlignHCenter)

        self.setFixedSize(600, 450)

        container = QWidget()
        container.setLayout(layout)
        container.setFixedSize(600, 450)
        self.setCentralWidget(container)

    def get_prediction(self):
        labels = {0: "Human-written", 1: "AI-generated"}
        text = self.input_textbox.toPlainText()
        try:
            result, confidence = predict(text, preprocessor.essay_processing_pipeline, threshold)
            self.output.setText((labels[result] + ". Confidence: " + str(confidence) + "%"))
        except ValueError:
            error_dialog = QErrorMessage(self)
            error_dialog.setWindowTitle("Error")
            error_dialog.showMessage("Error: Invalid input for model.")

# Load vocabulary
with open('../vocab.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

vocab = VocabGenerator(vocab=vocab_dict)

# Initialize essay preprocessor
preprocessor = EssayPreprocessor(vocab)

# Load model parameters
with open('../model-params.pkl', 'rb') as f:
    model_params = pickle.load(f)

vocab_size = model_params['vocab_size']
embed_size = model_params['embed_size']
hidden_size = model_params['hidden_size']
num_layers = model_params['num_layers']

# Define prediction threshold
threshold = 0.999693

# Set device to CPU
device = 'cpu'

# Load the model
model = EssayLSTM(vocab_size, embed_size, hidden_size, num_layers, device)
model.load_state_dict(torch.load('../ai-text-model.pt', weights_only=True))
model.eval()
model.to('cpu')

def predict(essay, essay_pipeline, threshold):
    with torch.no_grad():
        text = essay_pipeline(essay)
        text = torch.tensor(text)

        # Ensure input is valid for the model
        if len(text) == 0:
            raise ValueError("Invalid input for model.")

        sequence_length = torch.tensor(text.shape)
        # Reshape text for batch-first tensor format
        text = text.reshape(1, -1)
        output = model(text, sequence_length)
        result = (torch.sigmoid(output) >= threshold).int().item()
        print(output.item(), torch.sigmoid(output).item(), result)
        if result == 1:
            confidence = round((torch.sigmoid(output).item() - threshold) / (1-threshold) * 100, 1)
        else:
            confidence = round((threshold - torch.sigmoid(output).item()) / threshold * 100, 1)
        return result, confidence

app = QApplication([])

window = MainWindow()
window.show()

app.exec()
