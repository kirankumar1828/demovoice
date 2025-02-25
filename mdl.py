import json
import numpy as np
import tensorflow as tf
import nltk
import pickle
import os
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

# Ensure required downloads
# nltk.download("punkt")

# Load JSON data
with open("../voice/modell/data.json", "r") as file:
    data = json.load(file)

# Prepare training data
texts = []  # Input sentences
labels = []  # Corresponding intent labels

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())  # Convert to lowercase
        labels.append(intent["tag"])

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index  # Vocabulary dictionary
vocab_size = len(word_index) + 1  # Add 1 for padding

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)
max_length = max(len(seq) for seq in sequences)  # Find max length
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(set(encoded_labels))  # Unique label count
labels_categorical = tf.keras.utils.to_categorical(encoded_labels, num_classes)

# Define LSTM Model with Custom-Trained Embeddings
embedding_dim = 100  # Size of word vector representation

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

# Compile Model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
model.fit(padded_sequences, labels_categorical, epochs=50, batch_size=16, verbose=1, validation_split=0.1)

# Save Model and Required Files
model.save("intent_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("max_length.pkl", "wb") as f:
    pickle.dump(max_length, f)  # Save max_length for consistency

print("Training complete! Model saved.")
