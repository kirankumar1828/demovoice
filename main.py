import pyttsx3
import speech_recognition as sr
import datetime
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from nltk.tokenize import word_tokenize
from gmail import *
from apl import *
from system_operation import *
from browsing import *
from database import *

# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")

# Load trained intent classification model
model = tf.keras.models.load_model("../voice/modell/intent_model.h5")

# Load tokenizer and label encoder
with open("../voice/modell/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("../voice/modell/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

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


def listen_audio():
    """Continuously listen for audio commands."""
    try:
        while True:
            response = record()
            if response:
                main(response)
    except KeyboardInterrupt:
        return

def predict_intent(text):
    """Predict user intent using the trained deep learning model."""
    sequence = tokenizer.texts_to_sequences([text.lower()])
    padded_sequence = pad_sequences(sequence, maxlen=10, padding="post")
    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

def main(query):
    intent = predict_intent(query)  # Classify user intent
    done = False
    if intent == "greeting":
        speak("Hello! How can I assist you today?")
        done = True

    elif intent == "search_google":
        googleSearch(query)
        speak("Here are the results I found on Google.")
        done = True

    elif intent == "search_youtube":
        youtube(query)
        speak("Here are the results from YouTube.")
        done = True

    elif intent == "joke":
        joke = get_joke()
        if joke:
            speak(joke)
            done = True

    elif intent == "news":
        news = get_new()
        if news:
            speak(news)
            done = True
    
    elif intent == "ip" and "ip" in query: 
            ip = get_ip() 
            if ip: 
                speak(ip) 
                done = True

    elif intent == "get_time":
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The time is {current_time}")
        done = True

    elif intent == "get_date":
        current_date = datetime.datetime.now().strftime("%d %B, %Y")
        speak(f"Today's date is {current_date}")
        done = True

    elif intent == "get_datetime":
        current_datetime = datetime.datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
        speak(f"The current date and time is {current_datetime}")
        done = True

    elif intent == "weather":
        weather = get_weather()
        speak(f"The weather is: {weather}")
        done = True

    elif intent == "open_website":
        completed = open_specified_website(query)
        if completed:
            speak("Opening the website.")
            done = True

    elif intent == "email":
        speak("Please say the recipient's email address.")
        receiver_id = record()
        while not check_email(receiver_id):
            speak("Invalid email address. Please say it again.")
            receiver_id = record()
        
        speak("Say the subject of the email.")
        subject = record()
        
        speak("Say the body of the email.")
        body = record()
        
        success = send_email(receiver_id, subject, body)
        if success:
            speak("Your email has been sent successfully.")
        else:
            speak("There was an error sending the email.")
        done = True

    elif intent == "select_text" in query:
        sys_ops.select()
        speak("The text has been selected.")
        done = True
    elif intent == "copy_text" in query:
        sys_ops.copy()
        speak("The text has been copied.")
        done = True
    elif intent == "paste_text" in query:
        sys_ops.paste()
        speak("The text has been pasted.")
        done = True
    elif intent == "get_data" and "history" in query:
        get_data()
        done = True
    elif intent == "exit":
        speak("Thank you! Goodbye.")
        exit(0)
    

    if not done:
        answer = tell_me_about(query)
        if answer:
            speak(answer)
        else:
            speak("Sorry, I am not able to answer your query.")

    return

'''from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend requests

@app.route("/process_voice", methods=["POST"])
def process_voice():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"response": "I didn't hear anything. Please try again."})

    response_text = main(query)  # Ensure `main()` returns a response
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True)
'''

if __name__ == "__main__":
    listen_audio()
