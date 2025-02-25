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

def speak(text):
    """Convert text to speech."""
    print("ASSISTANT ->", text)
    try:
        engine.say(text)
        engine.runAndWait()
    except (KeyboardInterrupt, RuntimeError) as e:
        logger.error(f"Error in text-to-speech: {e}")

def record():
    """Record and recognize user speech."""
    with sr.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic)
        recognizer.dynamic_energy_threshold = True
        print("Listening...")
        try:
            audio = recognizer.listen(mic, timeout=5)
            text = recognizer.recognize_google(audio, language="en-US").lower()
            print("USER ->", text)
            return text
        except sr.UnknownValueError:
            logger.warning("Could not understand audio.")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return None
        except sr.WaitTimeoutError:
            logger.warning("Listening timed out.")
            return None

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

def main(query):
    """Process the user's command and execute the corresponding action."""
    if not query:
        speak("I didn't catch that. Please repeat.")
        return "No input detected."

    intent = predict_intent(query)
    if not intent:
        speak("Sorry, I couldn't understand your intent.")
        return "Intent prediction failed."

    intent_actions = {
        "greeting": lambda: speak("Hello! How can I assist you today?"),
        "search_google": lambda: (googleSearch(query), speak("Here are the results I found on Google.")),
        "search_youtube": lambda: (youtube(query), speak("Here are the results from YouTube.")),
        "joke": lambda: speak(get_joke() or "Sorry, I couldn't find a joke right now."),
        "news": lambda: speak(get_new() or "I'm unable to fetch news at the moment."),
        "ip": lambda: speak(get_ip() or "Couldn't retrieve IP address."),
        "get_time": lambda: speak(f"The time is {datetime.datetime.now().strftime('%I:%M %p')}"),
        "get_date": lambda: speak(f"Today's date is {datetime.datetime.now().strftime('%d %B, %Y')}"),
        "get_datetime": lambda: speak(f"The current date and time is {datetime.datetime.now().strftime('%A, %d %B %Y, %I:%M %p')}"),
        "weather": lambda: speak(f"The weather is: {get_weather()}"),
        "open_website": lambda: speak("Opening the website.") if open_specified_website(query) else speak("Unable to open website."),
        "select_text": lambda: (sys_ops.select(), speak("The text has been selected.")),
        "copy_text": lambda: (sys_ops.copy(), speak("The text has been copied.")),
        "paste_text": lambda: (sys_ops.paste(), speak("The text has been pasted.")),
        "get_data": lambda: get_data() if "history" in query else speak("I couldn't fetch the requested data."),
        "exit": lambda: (speak("Thank you! Goodbye."), exit(0)),
        "email": handle_email,
    }

    if intent in intent_actions:
        intent_actions[intent]()
        return "Intent executed successfully."

    # Default response if intent is not matched
    answer = tell_me_about(query)
    if answer:
        speak(answer)
        return answer
    else:
        speak("Sorry, I am not able to answer your query.")
        return "No answer found."

if __name__ == "__main__":
    listen_audio()