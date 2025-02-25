import pyttsx3
import speech_recognition as sr
import datetime
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import time

# Import modules for specific tasks
from gmail import *
from apl import *
from system_operation import *
from browsing import *
from database import *

# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")

# Load trained intent classification model
model = tf.keras.models.load_model("intent_model.h5")

# Load tokenizer and label encoder
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty("rate", 185)

sys_ops = SystemTasks()
tab_ops = TabOpt()
win_ops = WindowOpt()

def speak(text):
    """Convert text to speech."""
    print("ASSISTANT ->", text)
    try:
        engine.say(text)
        engine.runAndWait()
    except (KeyboardInterrupt, RuntimeError):
        return

def record():
    """Record and recognize user speech."""
    with sr.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic)
        recognizer.dynamic_energy_threshold = True
        print("Listening...")
        audio = recognizer.listen(mic)
        try:
            text = recognizer.recognize_google(audio, language="en-US").lower()
        except sr.UnknownValueError:
            return None
    print("USER ->", text)
    return text

def predict_intent(text):
    """Predict user intent using the trained deep learning model."""
    sequence = tokenizer.texts_to_sequences([text.lower()])
    padded_sequence = pad_sequences(sequence, maxlen=10, padding="post")
    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

def generate_response(query):
    """Generate and stream the response like ChatGPT."""
    intent = predict_intent(query)
    response_text = ""

    if intent == "greeting":
        response_text = "Hello! How can I assist you today?"

    elif intent == "search_google":
        googleSearch(query)
        response_text = "Here are the results I found on Google."

    elif intent == "search_youtube":
        youtube(query)
        response_text = "Here are the results from YouTube."

    elif intent == "joke":
        joke = get_joke()
        if joke:
            response_text = joke

    elif intent == "news":
        news = get_new()
        if news:
            response_text = news

    elif intent == "ip" and "ip" in query:
        ip = get_ip()
        if ip:
            response_text = ip

    elif intent == "get_time":
        response_text = f"The time is {datetime.datetime.now().strftime('%I:%M %p')}"

    elif intent == "get_date":
        response_text = f"Today's date is {datetime.datetime.now().strftime('%d %B, %Y')}"

    elif intent == "get_datetime":
        response_text = f"The current date and time is {datetime.datetime.now().strftime('%A, %d %B %Y, %I:%M %p')}"

    elif intent == "weather":
        weather = get_weather()
        response_text = f"The weather is: {weather}"

    elif intent == "open_website":
        completed = open_specified_website(query)
        if completed:
            response_text = "Opening the website."

    elif intent == "email":
        response_text = "Please say the recipient's email address."
        
    elif intent == "select_text":
        sys_ops.select()
        response_text = "The text has been selected."

    elif intent == "copy_text":
        sys_ops.copy()
        response_text = "The text has been copied."

    elif intent == "paste_text":
        sys_ops.paste()
        response_text = "The text has been pasted."

    elif intent == "get_data" and "history" in query:
        get_data()
        response_text = "Here is the requested data."

    elif intent == "exit":
        response_text = "Thank you! Goodbye."
        speak(response_text)
        exit(0)

    else:
        response_text = tell_me_about(query) or "Sorry, I am not able to answer your query."

    for word in response_text.split():
        yield word + " "
        time.sleep(0.2)

    speak(response_text)

@app.route("/process_voice", methods=["POST"])
def process_voice():
    """Handle voice input from the frontend and return a response."""
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"response": "I didn't hear anything. Please try again."})

    return Response(stream_with_context(generate_response(query)), content_type='text/event-stream')

def listen_audio():
    """Continuously listen for audio commands."""
    try:
        while True:
            response = record()
            if response:
                for _ in generate_response(response):
                    pass  # Stream responses if running without Flask
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    app.run(debug=True)
