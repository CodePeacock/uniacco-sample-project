# %pip install transformers
''''''
# Import necessary libraries
import random

import nltk
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

# from torch.utils.data import DataLoader

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# Load 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(
    subset="all",
    categories=["sci.space", "rec.sport.hockey", "talk.politics.guns", "rec.autos"],
)

# Create a pandas dataframe from the dataset
df = pd.DataFrame({"text": newsgroups.data, "label": newsgroups.target})

# Preprocess the text data
df["text"] = df["text"].str.lower()  # Lowercase text
df["text"] = df["text"].str.replace(r"[^\w\s]", "")  # Remove punctuation and digits
df["text"] = df["text"].str.replace(r"\d+", "")
df["text"] = df["text"].apply(word_tokenize)  # Tokenize text
stop_words = set(stopwords.words("english"))  # Remove stopwords
df["text"] = df["text"].apply(lambda x: [word for word in x if word not in stop_words])
df["text"] = df["text"].apply(lambda x: " ".join(x))  # Join tokens back into strings
df["text"] = df["text"].str.strip()  # Strip whitespace

# Split the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(train_df["label"].unique())
)

# Freeze the base BERT layers
for param in model.base_model.parameters():
    param.requires_grad = False

# Tokenize the text data for both the training and test sets
train_encodings = tokenizer(
    train_df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=512,  # Set the maximum sequence length to 512
)
test_encodings = tokenizer(
    test_df["text"].tolist(),
    truncation=True,
    padding=True,
    max_length=512,  # Set the maximum sequence length to 512
)

# Define a custom PyTorch dataset for the 20 Newsgroups dataset
class NewsGroupDataset(torch.utils.data.Dataset):
    """Custom PyTorch dataset for the 20 Newsgroups dataset"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
            if key != "overflowing_tokens"
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# Convert the tokenized data into PyTorch datasets
train_dataset = NewsGroupDataset(train_encodings, train_df["label"].tolist())
test_dataset = NewsGroupDataset(test_encodings, test_df["label"].tolist())

device = torch.device("cuda")

# Define the training arguments for the Trainer object
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    max_steps=1000,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,
)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=5e-5, eps=1e-8  # Increase learning rate
)

# Train the model
model = model.to(device)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    optimizers=(optimizer, None),
    compute_metrics=lambda pred: {
        "accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(axis=1)),
        "precision": precision_score(
            pred.label_ids, pred.predictions.argmax(axis=1), average="weighted"
        ),
        "recall": recall_score(
            pred.label_ids, pred.predictions.argmax(axis=1), average="weighted"
        ),
        "f1": f1_score(
            pred.label_ids, pred.predictions.argmax(axis=1), average="weighted"
        ),
    },
)


trainer.train()


eval_results = trainer.evaluate(test_dataset)
print(eval_results)

# Test the model on a random sample from the test set
sample_index = random.randint(0, len(test_df) - 1)
sample_text = test_df.iloc[sample_index]["text"]
sample_label = test_df.iloc[sample_index]["label"]
print("Sample text:", sample_text)
print("True label:", newsgroups.target_names[sample_label])


sample_encoding = tokenizer.encode_plus(
    sample_text, truncation=True, padding=True, return_tensors="pt"
)


model.eval()
with torch.no_grad():
    model_and_encoding = {"model": model, "encoding": sample_encoding}

    torch.save(model_and_encoding, "model_and_encoding.pt")


# Load the saved model and encoding
model_and_encoding = torch.load("model_and_encoding.pt")
saved_model = model_and_encoding["model"]
saved_encoding = model_and_encoding["encoding"]

# Move encoding tensor to the same device as the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
saved_encoding = {k: v.to(device) for k, v in saved_encoding.items()}

# Run inference
with torch.no_grad():
    output = saved_model(**saved_encoding)

predicted_label = output[0].argmax().item()
print("Predicted label:", newsgroups.target_names[predicted_label])
