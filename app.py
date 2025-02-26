from flask import Flask, request, jsonify
import pyttsx3
import speech_recognition as sr
import datetime
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
from gmail import *
from apl import *
from system_operation import *
from browsing import *
from database import *
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS


# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained intent classification model
model = tf.keras.models.load_model("intent_model.h5")

# Load tokenizer and label encoder
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty("rate", 185)
sys_ops = SystemTasks()

# Initialize Flask app
#app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process():
    """Process user input and return response."""
    data = request.json
    query = data.get("query")
    
    if not query:
        return jsonify({"action": "speak", "message": "I didn't catch that. Please repeat."})
    
    intent = predict_intent(query)
    
    response = handle_intent(intent, query)
    return jsonify(response)

def predict_intent(text):
    sequence = tokenizer.texts_to_sequences([text.lower()])
    print("Tokenized Sequence:", sequence)  # Debugging Line
    padded_sequence = pad_sequences(sequence, padding="post")  # Dynamic padding
    print("Padded Sequence Shape:", padded_sequence.shape)  # Debugging Line

    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

def handle_intent(intent, query):
    """Handle different intents and return a response."""
    if intent == "get_time":
        return {"action": "speak", "message": f"The time is {datetime.datetime.now().strftime('%I:%M %p')}"}
    elif intent == "greeting":
        return {"action": "speak", "message": "Hello! How can I assist you today?"}
    elif intent == "search_google":
        googleSearch(query)
        return {"action": "speak", "message": "Here are the results I found on Google."}
    elif intent == "search_youtube":
        youtube(query)
        return {"action": "speak", "message": "Here are the results from YouTube."}
    elif intent == "joke":
        return {"action": "speak", "message": get_joke() or "Sorry, I couldn't find a joke right now."}
    elif intent == "news":
        return {"action": "speak", "message": get_new() or "I'm unable to fetch news at the moment."}
    elif intent == "ip":
        return {"action": "speak", "message": get_ip() or "Couldn't retrieve IP address."}
    elif intent == "get_date":
        return {"action": "speak", "message": f"Today's date is {datetime.datetime.now().strftime('%d %B, %Y')}"}
    elif intent == "get_datetime":
        return {"action": "speak", "message": f"The current date and time is {datetime.datetime.now().strftime('%A, %d %B %Y, %I:%M %p')}"}
    elif intent == "weather":
        return {"action": "speak", "message": f"The weather is: {get_weather()}"}
    elif intent == "open_website":
        return {"action": "speak", "message": "Opening the website."} if open_specified_website(query) else {"action": "speak", "message": "Unable to open website."}
    elif intent == "select_text":
        sys_ops.select()
        return {"action": "speak", "message": "The text has been selected."}
    elif intent == "copy_text":
        sys_ops.copy()
        return {"action": "speak", "message": "The text has been copied."}
    elif intent == "paste_text":
        sys_ops.paste()
        return {"action": "speak", "message": "The text has been pasted."}
    elif intent == "get_data":
        return {"action": "speak", "message": "Fetching data."} if "history" in query else {"action": "speak", "message": "I couldn't fetch the requested data."}
    elif intent == "exit":
        return {"action": "speak", "message": "Thank you! Goodbye."}
    elif intent == "email":
        return handle_email()
    else:
        answer = tell_me_about(query)
        return {"action": "speak", "message": answer} if answer else {"action": "speak", "message": "Sorry, I am not able to answer your query."}

def handle_email():
    """Handle email-related tasks."""
    return {"action": "email", "message": "Please provide email details."}

if __name__ == "__main__":
    app.run(debug=True)
