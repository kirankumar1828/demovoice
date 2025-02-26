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

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()
engine.setProperty("rate", 185)

# Initialize system operations
sys_ops = SystemTasks()
tab_ops = TabOpt()
win_ops = WindowOpt()

def speak(text):
    """Convert text to speech."""
    print("ASSISTANT ->", text)
    try:
        engine.say(text)
        engine.runAndWait()
    except (KeyboardInterrupt, RuntimeError) as e:
        logger.error(f"Error in text-to-speech: {e}")
def record():
    with sr.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic)
        recognizer.dynamic_energy_threshold = True
        #recognizer.energy_threshold =50000
        print("Listening...")
        audio = recognizer.listen(mic)
        try:
            text = recognizer.recognize_google(audio, language='en-US').lower()
        except :
            return None
    print("USER -> " + text)
    return text

def listen_audio():
    """Continuously listen for audio commands."""
    try:
        while True:
            response = record()
            if response:
                main(response)
    except KeyboardInterrupt:
        logger.info("Exiting...")
        return

def predict_intent(text):
    sequence = tokenizer.texts_to_sequences([text.lower()])
    print("Tokenized Sequence:", sequence)  # Debugging Line
    padded_sequence = pad_sequences(sequence, padding="post")  # Dynamic padding
    print("Padded Sequence Shape:", padded_sequence.shape)  # Debugging Line

    prediction = model.predict(padded_sequence)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]


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

def get_time():
    """Provide the current time."""
    current_time = datetime.datetime.now().strftime('%I:%M %p')
    speak(f"The time is {current_time}")

def main(query):
    """Process the user's command and execute the corresponding action."""
    if not query:
        speak("I didn't catch that. Please repeat.")
        return "No input detected."

    intent = predict_intent(query)

    # Mapping intents to functions
    if intent == "get_time":
        get_time()
    elif intent == "greeting":
        speak("Hello! How can I assist you today?")
    elif intent == "search_google":
        googleSearch(query)
        speak("Here are the results I found on Google.")
    elif intent == "search_youtube":
        youtube(query)
        speak("Here are the results from YouTube.")
    elif intent == "joke":
        joke = get_joke()
        speak(joke if joke else "Sorry, I couldn't find a joke right now.")
    elif intent == "news":
        news = get_new()
        speak(news if news else "I'm unable to fetch news at the moment.")
    elif intent == "ip":
        ip_address = get_ip()
        speak(ip_address if ip_address else "Couldn't retrieve IP address.")
    elif intent == "get_date":
        current_date = datetime.datetime.now().strftime('%d %B, %Y')
        speak(f"Today's date is {current_date}")
    elif intent == "get_datetime":
        date_time = datetime.datetime.now().strftime('%A, %d %B %Y, %I:%M %p')
        speak(f"The current date and time is {date_time}")
    elif intent == "weather":
        weather_info = get_weather()
        speak(f"The weather is: {weather_info}")
    elif intent == "open_website":
        if open_specified_website(query):
            speak("Opening the website.")
        else:
            speak("Unable to open website.")
    elif intent == "select_text":
        sys_ops.select()
        speak("The text has been selected.")
    elif intent == "copy_text":
        sys_ops.copy()
        speak("The text has been copied.")
    elif intent == "paste_text":
        sys_ops.paste()
        speak("The text has been pasted.")
    elif intent == "get_data":
        if "history" in query:
            get_data()
        else:
            speak("I couldn't fetch the requested data.")
    elif intent == "exit":
        speak("Thank you! Goodbye.")
        exit(0)
    elif intent == "email":
        handle_email()
    else:
        answer = tell_me_about(query)
        if answer:
            speak(answer)
            return answer
        else:
            speak("Sorry, I am not able to answer your query.")
            return "No answer found."

    return "Intent executed successfully."

if __name__ == "__main__":
    listen_audio()
