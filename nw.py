from flask import Flask, request, jsonify, Response
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
with open("max_length.pkl", "rb") as f:
    max_length = pickle.load(f)  # Ensure consistent input shape

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty("rate", 185)

# Initialize system operations
sys_ops = SystemTasks()
tab_ops = TabOpt()
win_ops = WindowOpt()

# Flask app
app = Flask(__name__)

def predict_intent(text):
    """Predict user intent using the trained deep learning model."""
    try:
        sequence = tokenizer.texts_to_sequences([text.lower()])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post")
        prediction = model.predict(padded_sequence)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_label[0]
    except Exception as e:
        logger.error(f"Error predicting intent: {e}")
        return None

def handle_email():
    """Handle email-related tasks."""
    speak("Please say the recipient's email address.")
    receiver_id = sanitize_email(record())

    while not receiver_id:
        speak("Invalid email address. Please say it again.")
        receiver_id = sanitize_email(record())

    speak("Say the subject of the email.")
    subject = record() or "No Subject"

    speak("Say the body of the email.")
    body = record() or "No Content"

    if send_email(receiver_id, subject, body):
        speak("Your email has been sent successfully.")
    else:
        speak("There was an error sending the email.")

def process_query(query):
    """Process the user's query and return the assistant's response."""
    if not query:
        return "I didn't catch that. Please repeat."

    intent = predict_intent(query)
    if not intent:
        return "Sorry, I couldn't understand your intent."

    intent_actions = {
        "greeting": lambda: "Hello! How can I assist you today?",
        "search_google": lambda: googleSearch(query) or "Here are the results I found on Google.",
        "search_youtube": lambda: youtube(query) or "Here are the results from YouTube.",
        "joke": lambda: get_joke() or "Sorry, I couldn't find a joke right now.",
        "news": lambda: get_new() or "I'm unable to fetch news at the moment.",
        "ip": lambda: get_ip() or "Couldn't retrieve IP address.",
        "get_time": lambda: f"The time is {datetime.datetime.now().strftime('%I:%M %p')}",
        "get_date": lambda: f"Today's date is {datetime.datetime.now().strftime('%d %B, %Y')}",
        "get_datetime": lambda: f"The current date and time is {datetime.datetime.now().strftime('%A, %d %B %Y, %I:%M %p')}",
        "weather": lambda: f"The weather is: {get_weather()}",
        "open_website": lambda: "Opening the website." if open_specified_website(query) else "Unable to open website.",
        "select_text": lambda: (sys_ops.select(), "The text has been selected."),
        "copy_text": lambda: (sys_ops.copy(), "The text has been copied."),
        "paste_text": lambda: (sys_ops.paste(), "The text has been pasted."),
        "get_data": lambda: get_data() if "history" in query else "I couldn't fetch the requested data.",
        "exit": lambda: (speak("Thank you! Goodbye."), exit(0)),
        "email": handle_email,
    }

    if intent in intent_actions:
        return intent_actions[intent]()
    else:
        answer = tell_me_about(query)
        return answer if answer else "Sorry, I am not able to answer your query."

@app.route("/process_voice", methods=["POST"])
def process_voice():
    """Handle voice input from the frontend and stream the response."""
    data = request.json
    query = data.get("query", "")

    def generate():
        response = process_query(query)
        for word in response.split():
            yield f"data: {word}\n\n"
            import time
            time.sleep(0.1)  # Simulate streaming
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(debug=True)