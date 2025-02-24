import json
import numpy as np
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, Attention
from tensorflow.keras.models import Sequential
import pickle

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

# Tokenize words
#nltk.download("punkt")
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")  # Increased vocab size
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=15, padding="post")  # Increased maxlen

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(set(encoded_labels))

# Convert labels to categorical format
labels_categorical = tf.keras.utils.to_categorical(encoded_labels, num_classes)

# Define improved LSTM-based model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=15),  # Increased embedding size
    Bidirectional(LSTM(128, return_sequences=True)),  # Bidirectional LSTM
    Bidirectional(LSTM(64)),  
    Dense(64, activation="relu"),
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(num_classes, activation="softmax")
])

# Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(padded_sequences, labels_categorical, epochs=100, batch_size=16, verbose=1)  # Increased epochs

# Save model and tokenizer
model.save("intent_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Training complete! Model saved.")
