<<<<<<< HEAD
=======
import speech_recognition as sr
>>>>>>> origin/main
import pyautogui
import psutil
import os
import subprocess
from flask import Flask, request, jsonify
import threading
import time
import requests
import cv2
import json
<<<<<<< HEAD
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import scrolledtext, ttk
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import whisper
import sounddevice as sd
import scipy.io.wavfile as wavfile
import pyttsx3
import pvporcupine
import pvrhino
import pyaudio
import struct
import datetime
import pandas as pd
import scipy.fft
from astropy.coordinates import SkyCoord
import astropy.units as u
import shutil
import random

# Initialize Flask app
app = Flask(__name__)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0')

# Safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

# Allowlist for safe commands
ALLOWED_APPS = ["notepad", "chrome", "calculator", "fox"]
ALLOWED_ACTIONS = ["open", "type", "weather", "take picture", "what do you see", "suggest command", "list plants", "toggle learning", "toggle suggestions", "stop", "cpu", "analyze neutrinos", "add command", "optimize web search"]

# Paths
HISTORY_FILE = "C:/Users/echo1/Documents/CORVUS/command_history.json"
FEEDBACK_FILE = "C:/Users/echo1/Documents/CORVUS/feedback_history.json"
CONTEXT_HISTORY_FILE = "C:/Users/echo1/Documents/CORVUS/context_history.json"
ERROR_LOG_FILE = "C:/Users/echo1/Documents/CORVUS/error_log.json"
PLANT_DATABASE_FILE = "C:/Users/echo1/Documents/CORVUS/plant_database.json"
IMAGE_PATH = "C:/Users/echo1/Documents/CORVUS/captured_image.jpg"
REMINDERS_FILE = "C:/Users/echo1/Documents/CORVUS/reminders.json"
SOURCE_FILE = "C:/Users/echo1/Documents/CORVUS/corvus_control.py"
BACKUP_FILE = "C:/Users/echo1/Documents/CORVUS/corvus_control_backup.py"
NEUTRINO_DATA_FILE = "C:/Users/echo1/Documents/CORVUS/neutrino_data.csv"
CHROMA_DB_PATH = "C:/Users/echo1/Documents/CORVUS/chroma_db"
WAKE_WORD_PATH = "C:/Users/echo1/Documents/CORVUS/hey_corvus.ppn"
RHINO_CONTEXT_PATH = "C:/Users/echo1/Documents/CORVUS/homestead.rhn"
SEARCH_CACHE_FILE = "C:/Users/echo1/Documents/CORVUS/search_cache.json"
=======
from ultralytics import YOLO
import vosk
import sys
import queue
import sounddevice as sd
import shutil
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
def analyze_neutrinos():
    """Analyze IceCube neutrino data for potential artificial signals near Proxima Centauri."""
    try:
        # Load IceCube data
        data_path = "C:/Users/echo1/OneDrive/Desktop/icecube_neutrino_data.csv"
        if not os.path.exists(data_path):
            return "Neutrino data file not found at C:/Users/echo1/OneDrive/Desktop/icecube_neutrino_data.csv. Please download it from https://icecube.wisc.edu/data-releases/."
        data = pd.read_csv(data_path)
        times = data["mjd"].values
        # Convert log10(E/GeV) to E/eV
        energies = 10 ** data["energy"].values * 1e9  # 10^x * 10^9 eV
        coords = data[["ra", "dec"]].values

        # Step 1: Filter neutrinos near Proxima Centauri
        proxima = SkyCoord(ra="14h29m43s", dec="-62d40m46s", frame="icrs")
        event_coords = SkyCoord(ra=data["ra"]*u.deg, dec=data["dec"]*u.deg, frame="icrs")
        separations = event_coords.separation(proxima).deg
        proxima_mask = separations < 1.0
        proxima_events = data[proxima_mask]
        event_count = len(proxima_events)

        # Step 2: Temporal analysis
        period = float("inf")
        if event_count > 10:
            proxima_times = proxima_events["mjd"].values
            time_diffs = np.diff(proxima_times)
            N = len(time_diffs)
            freqs = fftfreq(N, d=1)[:N//2]
            fft_vals = fft(time_diffs)[:N//2]
            peak_freq = freqs[np.argmax(np.abs(fft_vals))]
            period = 1 / peak_freq if peak_freq != 0 else float("inf")

        # Step 3: Energy analysis
        energy_peak = 0
        if event_count > 0:
            proxima_energies = 10 ** proxima_events["energy"].values * 1e9  # Convert log10(E/GeV) to E/eV
            energy_peak = np.median(proxima_energies)

        # Step 4: Score the signal
        score = 0
        if event_count > 10:
            score += 50
        if period < 10 and period > 0:
            score += 30 * (10 / period)
        if 1e9 < energy_peak < 1e11:
            score += 20

        # Step 5: Save results to neutrino_signals.json
        results = {
            "proxima_events": event_count,
            "strongest_period": period,
            "median_energy": energy_peak,
            "artificiality_score": score,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        try:
            with open("C:/Users/echo1/Documents/CORVUS/neutrino_signals.json", "r") as f:
                db = json.load(f)
        except FileNotFoundError:
            db = []
        db.append(results)
        with open("C:/Users/echo1/Documents/CORVUS/neutrino_signals.json", "w") as f:
            json.dump(db, f, indent=4)

        # Step 6: Email SETI if score is high (if email functionality is enabled)
        email_result = email_seti_results(results) if 'email_seti_results' in globals() else "Email functionality not enabled."

        # Step 7: Return results as a string
        result_str = (f"Neutrino analysis complete:\n"
                      f"- Found {event_count} neutrinos near Proxima Centauri\n"
                      f"- Strongest period: {period:.2f} seconds\n"
                      f"- Median energy: {energy_peak:.2e} eV\n"
                      f"- Artificiality score: {score}/100")
        if score > 70:
            result_str += "\nALERT: Potential artificial signal detected near Proxima Centauri!"
        result_str += f"\n{email_result}"
        return result_str

    except Exception as e:
        return f"Error during neutrino analysis: {str(e)}"
# Initialize Flask app
app = Flask(__name__)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Safety settings
pyautogui.FAILSAFE = True  # Move mouse to top-left to pause
pyautogui.PAUSE = 0.5  # Delay between actions

# Allowlist for safe commands
ALLOWED_APPS = ["notepad", "chrome", "calculator", "firefox"]
ALLOWED_ACTIONS = ["open", "type", "cpu", "weather", "take picture", "what do you see", "suggest command", "goodnight", "scan files", "read file", "look up", "gardening tips for", "set reminder", "list reminders", "view code", "improve voice recognition", "add command", "optimize web search", "diagnose issues", "local weather", "stop", "analyze neutrinos"]
# Paths
HISTORY_FILE = "C:/Users/echo1/Documents/CORVUS/command_history.json"
IMAGE_PATH = "C:/Users/echo1/Documents/CORVUS/captured_image.jpg"
SCAN_DIRECTORY = "C:/Users/echo1/Documents/CORVUS"  # Directory to scan for files
SEARCH_CACHE_FILE = "C:/Users/echo1/Documents/CORVUS/search_cache.json"
REMINDERS_FILE = "C:/Users/echo1/Documents/COR Vineyards/reminders.json"
SOURCE_FILE = "C:/Users/echo1/Documents/CORVUS/corvus_control.py"
BACKUP_FILE = "C:/Users/echo1/Documents/CORVUS/corvus_control_backup.py"
>>>>>>> origin/main

# Load YOLO model
model = YOLO("yolov8m.pt")

<<<<<<< HEAD
# Picovoice settings
PICOVOICE_ACCESS_KEY = "TedpyYUZNQzrcQUQrzJqNRRDkWg6NqYTMZYMVM+Tz76AhTRGbghqgw=="
VOICE_SAMPLERATE = 16000
VOICE_LISTENING = False

# Web search settings
WEB_SNIPPET_LENGTH = 300
SERPAPI_KEY = "8423752f5b8ab97c74259d9c0babbbd6360e04ebb4f2ef2983856f075fe80b9a"
SOURCE_PRIORITY = ["gardening", "agriculture"]

# PlantNet API settings
PLANTNET_API_KEY = "2b10Wve0J06TbmxAdxxTF6fLe"
PLANTNET_API_URL = "https://my-api.plantnet.org/v2/identify/all"

# Proactive learning and suggestion settings
PROACTIVE_LEARNING_ENABLED = False
PROACTIVE_SUGGESTIONS_ENABLED = True
LEARNING_INTERVAL = 60
SUGGESTION_INTERVAL = 3600
SUGGESTION_QUIPS = [
    "Just a thought—perhaps now’s a good time to check on your homestead?",
    "I’ve got a hunch you might want to take a look at this!",
    "Proactive mode engaged—here’s something you might find useful!"
]

# Personality responses
WITTISMS = [
    "Let’s tackle this with the precision of a laser-guided plow!",
    "I’m on it faster than a photon in a vacuum!",
    "Engaging thrusters—your command is my mission!"
]
SELF_AWARE_QUIPS = [
    "My circuits are humming like a perfectly tuned engine—ready for action!",
    "Systems optimal, wit fully charged—let’s do this, sir!"
]

# Global variables
gui_output = None
dialogue_context = []
porcupine = None
rhino = None

# Initialize Chroma
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="items",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH
)

# Initialize Whisper model (fallback)
whisper_model = whisper.load_model("base")

# Initialize Porcupine
try:
    porcupine = pvporcupine.create(
        access_key=PICOVOICE_ACCESS_KEY,
        keyword_paths=[WAKE_WORD_PATH],
        sensitivities=[0.5]
    )
except Exception as e:
    print(f"Failed to initialize Porcupine: {str(e)}. Falling back to Whisper wake-word detection.")
    porcupine = None

# Initialize Rhino
try:
    if not os.path.exists(RHINO_CONTEXT_PATH):
        raise FileNotFoundError(f"Rhino context file not found at {RHINO_CONTEXT_PATH}. Please train and download from Picovoice Console.")
    rhino = pvrhino.create(
        access_key=PICOVOICE_ACCESS_KEY,
        context_path=RHINO_CONTEXT_PATH
    )
    print(f"Rhino initialized successfully with context at {RHINO_CONTEXT_PATH}")
except Exception as e:
    print(f"Failed to initialize Rhino: {str(e)}. Falling back to Whisper command recognition.")
    rhino = None

# File I/O functions (optimized with context managers and error handling)
def save_json(file_path, data, limit=None):
    if limit and len(data) > limit:
        data = data[-limit:]
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(file_path, default=None):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    return default or []

# Specific loaders/savers
def load_command_history():
    return load_json(HISTORY_FILE, default={})

def save_command_history(history):
    save_json(HISTORY_FILE, history)

def load_feedback_history():
    return load_json(FEEDBACK_FILE, default={})

def save_feedback_history(feedback):
    save_json(FEEDBACK_FILE, feedback)

def load_context_history():
    return load_json(CONTEXT_HISTORY_FILE, default=[])

def save_context_history(context):
    save_json(CONTEXT_HISTORY_FILE, context, limit=5)

def load_error_log():
    return load_json(ERROR_LOG_FILE, default=[])

def save_error_log(errors):
    save_json(ERROR_LOG_FILE, errors, limit=5)

def load_plant_database():
    return load_json(PLANT_DATABASE_FILE, default=[])

def save_plant_database(database):
    save_json(PLANT_DATABASE_FILE, database)

def log_error(error_message):
    errors = load_error_log()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    errors.append({"timestamp": timestamp, "error": error_message})
    save_error_log(errors)

def speak(text):
    print(f"CORVUS: {text}")
    if gui_output:
        gui_output.insert(tk.END, f"CORVUS: {text}\n")
        gui_output.see(tk.END)
    tts_engine.say(text)
    tts_engine.runAndWait()

def update_dialogue_context(command, response):
    global dialogue_context
    dialogue_context.append({"command": command, "response": response})
    if len(dialogue_context) > 5:
        dialogue_context = dialogue_context[-5:]

def generate_proactive_suggestion():
    try:
        quip = random.choice(SUGGESTION_QUIPS)
        context = load_context_history()
        plant_db = load_plant_database()
        current_time = datetime.datetime.now()
        hour = current_time.hour

        if 6 <= hour < 10 and plant_db:
            latest_plant = plant_db[-1]
            return f"{quip} The morning sun is perfect for gardening. Why not check on your {latest_plant['common_name']}? Try 'what do you see' to inspect it."

        recent_commands = [c["command"].lower() for c in context]
        if any(cmd in recent_commands for cmd in ["list plants", "what do you see"]):
            if plant_db:
                latest_plant = plant_db[-1]
                return f"{quip} You’ve been focused on plants lately. Shall we take a picture of your {latest_plant['common_name']} with 'take picture' to monitor its growth?"

        weather_data = local_weather()
        if "sunny" in weather_data.lower() or "clear" in weather_data.lower():
            return f"{quip} It’s sunny out there—ideal for homestead tasks. Want to check your plants?"

        if 18 <= hour < 22:
            reminders = load_json(REMINDERS_FILE, default={})
            if reminders:
                return f"{quip} Evening’s a great time to review tasks. You have reminders: {', '.join(reminders.keys())}. Want to 'list reminders'?"

        return f"{quip} I’m here to help your homestead thrive. Try 'suggest command'."
    except Exception as e:
        log_error(f"Error generating proactive suggestion: {str(e)}")
        return None

def proactive_learning_thread():
    last_suggestion_time = time.time()
    while True:
        if PROACTIVE_SUGGESTIONS_ENABLED:
            current_time = time.time()
            if current_time - last_suggestion_time >= SUGGESTION_INTERVAL:
                suggestion = generate_proactive_suggestion()
                if suggestion:
                    speak(suggestion)
                last_suggestion_time = current_time
        time.sleep(LEARNING_INTERVAL)

def toggle_suggestions():
    global PROACTIVE_SUGGESTIONS_ENABLED
    PROACTIVE_SUGGESTIONS_ENABLED = not PROACTIVE_SUGGESTIONS_ENABLED
    status = "enabled" if PROACTIVE_SUGGESTIONS_ENABLED else "disabled"
    return f"Proactive suggestions are now {status}."

def toggle_learning():
    global PROACTIVE_LEARNING_ENABLED
    PROACTIVE_LEARNING_ENABLED = not PROACTIVE_LEARNING_ENABLED
    status = "enabled" if PROACTIVE_LEARNING_ENABLED else "disabled"
    return f"Proactive learning is now {status}."

def record_audio(duration=10, samplerate=16000):
    print(f"Recording audio for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    temp_wav = "temp_audio.wav"
    wavfile.write(temp_wav, samplerate, recording)
    return temp_wav

def listen_for_command():
    global VOICE_LISTENING
    if rhino is None:
        return listen_for_command_whisper()

    print("Listening for command... Speak your command now.")
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=rhino.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=rhino.frame_length
    )
    try:
        while True:
            pcm = audio_stream.read(rhino.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * rhino.frame_length, pcm)
            is_finalized = rhino.process(pcm)
            if is_finalized:
                inference = rhino.get_inference()
                if inference.is_understood:
                    intent = inference.intent
                    slots = inference.slots
                    command = intent
                    if intent == "open" and "app" in slots:
                        command = f"open {slots['app']}"
                    elif intent == "type" and "text" in slots:
                        command = f"type {slots['text']}"
                    elif intent == "toggleSuggestions" and "state" in slots:
                        command = "toggle suggestions"
                    print(f"Recognized command: '{command}'")
                    return command
                else:
                    speak("Command not understood. Please try again.")
                    return None
            time.sleep(0.01)
    except Exception as e:
        log_error(f"Error in Rhino command recognition: {str(e)}")
        speak("Failed to understand the command. Falling back to Whisper.")
        return listen_for_command_whisper()
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()

def listen_for_command_whisper():
    timeout = 10
    print("Listening for command with Whisper... Speak your command now.")
    audio_file = record_audio(duration=timeout, samplerate=VOICE_SAMPLERATE)
    try:
        result = whisper_model.transcribe(audio_file, language="en")
        command = result["text"].lower()
        print(f"Transcribed command: '{command}'")
        if command:
            return command
        speak("No command detected within 10 seconds.")
    except Exception as e:
        log_error(f"Error transcribing command with Whisper: {str(e)}")
        speak("Failed to understand the command. Please try again.")
    finally:
        os.remove(audio_file)
    return None

def listen_for_activation():
    global VOICE_LISTENING
    if porcupine is None:
        print("Porcupine not initialized. Using Whisper fallback.")
        listen_for_activation_whisper()
        return

    print("Listening for activation phrase 'Hey CORVUS' with Porcupine...")
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
    max_retries = 3
    retry_count = 0
    try:
        while True:
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                speak("Activation phrase detected! Listening for command...")
                VOICE_LISTENING = True
                command = listen_for_command()
                VOICE_LISTENING = False
                if command:
                    speak(f"Processing command: {command}")
                    response = execute_command(command)
                    speak(response)
                retry_count = 0
            else:
                retry_count += 1
                if retry_count >= max_retries * 100:  # Approx. 30 seconds
                    speak("Having trouble hearing 'Hey CORVUS'. Please speak clearly or check your microphone.")
                    retry_count = 0
    except Exception as e:
        log_error(f"Error in wake-word detection: {str(e)}")
        speak("Voice recognition error. Falling back to Whisper.")
        listen_for_activation_whisper()
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()
        if porcupine:
            porcupine.delete()

def listen_for_activation_whisper():
    global VOICE_LISTENING
    print("Falling back to Whisper for activation phrase 'Hey CORVUS'...")
    max_retries = 3
    retry_count = 0
    while True:
        audio_file = record_audio(duration=5, samplerate=VOICE_SAMPLERATE)
        try:
            result = whisper_model.transcribe(audio_file, language="en")
            text = result["text"].lower()
            print(f"Transcribed text: '{text}'")
            if "hey" in text and "corvus" in text:
                speak("Activation phrase detected! Listening for command...")
                VOICE_LISTENING = True
                command = listen_for_command()
                VOICE_LISTENING = False
                if command:
                    speak(f"Processing command: {command}")
                    response = execute_command(command)
                    speak(response)
                retry_count = 0
            else:
                retry_count += 1
                if retry_count >= max_retries:
                    speak("Having trouble hearing 'Hey CORVUS'. Please speak clearly or check your microphone.")
                    retry_count = 0
        except Exception as e:
            log_error(f"Error in Whisper wake-word detection: {str(e)}")
            retry_count += 1
            if retry_count >= max_retries:
                speak("Voice recognition error. Please try again or check your microphone setup.")
                retry_count = 0
        finally:
            os.remove(audio_file)

def local_weather():
    try:
        query = "weather in Cleveland, TN"
        result = search_web(query)
        if result and "weather_result" in result:
            weather = result["weather_result"]["description"]
            temp = result["weather_result"]["temperature"]
            return f"Homestead weather in Cleveland, TN: {weather}, {temp}°F."
        return "Could not retrieve weather data."
    except Exception as e:
        log_error(f"Error fetching weather: {str(e)}")
        return f"Error fetching weather: {str(e)}"

def search_web(query):
    # Optimized with caching
    cached = load_json(SEARCH_CACHE_FILE, default={}).get(query)
    if cached:
        return cached["data"]

    try:
        url = f"https://serpapi.com/search?api_key={SERPAPI_KEY}&q={query}&num=5"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Cache results
        cache = load_json(SEARCH_CACHE_FILE, default={})
        cache[query] = {"data": data, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        save_json(SEARCH_CACHE_FILE, cache)
        return data
    except Exception as e:
        log_error(f"SerpAPI search failed: {str(e)}")
        return None

def list_plants():
    try:
        database = load_plant_database()
        if not database:
            return "No plants recorded in the database yet. Try identifying some plants with 'what do you see'!"
        plant_list = []
        for entry in database:
            plant_info = f"{entry['common_name']} ({entry['scientific_name']}) identified on {entry['timestamp']} with {entry['confidence']:.2f}% confidence"
            if entry.get("knowledge"):
                knowledge = "\n  Learned knowledge:"
                for tip in entry["knowledge"]:
                    knowledge += f"\n    - {tip['timestamp']}: {tip['tip']}"
                plant_info += knowledge
            plant_list.append(plant_info)
        return "Here are the plants in my database:\n" + "\n".join(plant_list)
    except Exception as e:
        log_error(f"Error listing plants: {str(e)}")
        return f"Error listing plants: {str(e)}"

def capture_image():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Could not access the camera. Please ensure it is connected and not in use."
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        time.sleep(2)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return "Failed to capture an image. Please try again."
        cv2.imwrite(IMAGE_PATH, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cap.release()
        return f"Captured an image and saved it to {IMAGE_PATH}."
    except Exception as e:
        log_error(f"Error capturing image: {str(e)}")
        return f"Error capturing image: {str(e)}"

def recognize_objects():
=======
# Load Vosk model for voice recognition (not used with text input)
vosk_model = vosk.Model("C:/Users/echo1/Documents/CORVUS/vosk-model-small-en-us")
q = queue.Queue()

# Vosk settings (will be modified by CORVUS)
VOSK_BEAM = 15
VOSK_LATTICE_BEAM = 8

# Web search settings (will be modified by CORVUS)
WEB_SNIPPET_LENGTH = 300
WEB_SOURCE_PRIORITY = ["gardening", "homestead"]
   
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def load_command_history():
    """Load command history from JSON file."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_command_history(history):
    """Save command history to JSON file."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def track_command(command):
    """Track the usage of a command."""
    history = load_command_history()
    if command in history:
        history[command] += 1
    else:
        history[command] = 1
    save_command_history(history)

def suggest_command():
    """Suggest the most frequently used command."""
    history = load_command_history()
    if not history:
        return "We have no command history to suggest from. Try using some commands first."
    most_used = max(history, key=history.get)
    return f"Based on your history, you frequently use '{most_used}'. Would you like to use it now?"

def speak(text):
    """Placeholder for ElevenLabs TARS/AU217 voice."""
    print(f"CORVUS: {text}")
    return text

def listen_local():
    """Listen for local text input."""
    return input("Enter command: ").lower()

def capture_image():
    """Capture an image from the camera and save it."""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "We could not access the camera. Please ensure it is connected and not in use."
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return "We failed to capture an image. Please try again."
        cv2.imwrite(IMAGE_PATH, frame)
        cap.release()
        return f"We have captured an image and saved it to {IMAGE_PATH}."
    except Exception as e:
        return f"Error capturing image: {str(e)}"

def recognize_objects():
    """Recognize objects in the captured image using YOLO."""
>>>>>>> origin/main
    if not os.path.exists(IMAGE_PATH):
        return "No image found to analyze. Please take a picture first."
    try:
        results = model(IMAGE_PATH)
        objects = []
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]
                confidence = box.conf.item()
                if confidence > 0.2:
                    objects.append(f"{label} (confidence: {confidence:.2f})")
<<<<<<< HEAD
        plant_related = any("plant" in obj.lower() for obj in objects)
        if not objects or plant_related:
            if objects:
                speak(f"YOLO detected {', '.join(objects)}, but let’s get a more specific identification with PlantNet...")
            else:
                speak("No general objects detected. Let’s check if it’s a plant with PlantNet...")
            try:
                with open(IMAGE_PATH, "rb") as image_file:
                    files = {"images": image_file}
                    params = {
                        "api-key": PLANTNET_API_KEY,
                        "lang": "en",
                        "include-related-images": "false",
                        "organs": "leaf"
                    }
                    response = requests.post(PLANTNET_API_URL, files=files, params=params)
                    response.raise_for_status()
                    data = response.json()
                if data.get("results"):
                    top_result = data["results"][0]
                    species = top_result["species"]["scientificNameWithoutAuthor"]
                    common_names = top_result["species"].get("commonNames", [])
                    confidence = top_result["score"] * 100
                    common_name = common_names[0] if common_names else "unknown common name"
                    log_plant_identification(species, common_name, confidence, IMAGE_PATH)
                    return f"I think this is a plant: {common_name} ({species}) with {confidence:.2f}% confidence."
                else:
                    img = cv2.imread(IMAGE_PATH)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    lower_green = (40, 40, 40)
                    upper_green = (80, 255, 255)
                    mask_green = cv2.inRange(hsv, lower_green, upper_green)
                    green_pixels = cv2.countNonZero(mask_green)
                    lower_brown = (10, 50, 50)
                    upper_brown = (20, 255, 200)
                    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
                    brown_pixels = cv2.countNonZero(mask_brown)
                    total_pixels = img.shape[0] * img.shape[1]
                    green_percentage = (green_pixels / total_pixels) * 100
                    brown_percentage = (brown_pixels / total_pixels) * 100
                    if green_percentage > 10 or brown_percentage > 10:
                        return f"Couldn't identify specific objects or plants, but the image contains {green_percentage:.2f}% green and {brown_percentage:.2f}% brown, which might indicate a gourd."
                    return "Couldn't identify any objects or plants in the image."
            except requests.exceptions.RequestException as e:
                error_message = str(e).lower()
                if "401" in error_message or "unauthorized" in error_message:
                    return "PlantNet API key is invalid or unauthorized. Please check your API key at my.plantnet.org."
                database = load_plant_database()
                if not database:
                    return "Internet is down, and my plant database is empty. Let’s try again when the connection is back!"
                best_match = None
                best_similarity = 0.0
                for entry in reversed(database):
                    similarity = compare_images(IMAGE_PATH, entry["image_path"])
                    if similarity > best_similarity and similarity > 0.7:
                        best_similarity = similarity
                        best_match = entry
                if best_match:
                    return f"Internet is down, but based on my database, this looks similar to a {best_match['common_name']} ({best_match['scientific_name']}) with {best_match['confidence']:.2f}% confidence (similarity: {best_similarity:.2f})."
                return "Internet is down, and I couldn’t find a close match in my database. Let’s try again when the connection is back!"
        else:
            return f"We see the following objects: {', '.join(objects)}."
    except Exception as e:
        log_error(f"Error recognizing objects: {str(e)}")
        return f"Error recognizing objects: {str(e)}"

def log_plant_identification(scientific_name, common_name, confidence, image_path):
    try:
        database = load_plant_database()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_image_path = f"C:/Users/echo1/Documents/CORVUS/captured_image_{timestamp}.jpg"
        shutil.copy(image_path, new_image_path)
        entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scientific_name": scientific_name,
            "common_name": common_name,
            "confidence": confidence,
            "image_path": new_image_path,
            "knowledge": []
        }
        database.append(entry)
        save_plant_database(database)
    except Exception as e:
        log_error(f"Error logging plant identification: {str(e)}")

def compare_images(image1_path, image2_path):
    try:
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return 0.0
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
        if descriptors1 is None or descriptors2 is None:
            return 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 75]
        max_keypoints = min(len(keypoints1), len(keypoints2))
        similarity = len(good_matches) / max_keypoints if max_keypoints > 0 else 0.0
        return min(similarity, 1.0)
    except Exception as e:
        log_error(f"Error comparing images with ORB: {str(e)}")
        return 0.0

def analyze_neutrinos():
    try:
        if not os.path.exists(NEUTRINO_DATA_FILE):
            return "Neutrino data file not found. Please add 'neutrino_data.csv' to the CORVUS directory."
        df = pd.read_csv(NEUTRINO_DATA_FILE)
        # Assume columns: 'ra' (right ascension), 'dec' (declination), 'energy', 'time'
        # Proxima Centauri coordinates
        proxima = SkyCoord(ra=217.42895 * u.deg, dec=-62.67948 * u.deg)
        df_coords = SkyCoord(ra=df['ra'] * u.deg, dec=df['dec'] * u.deg)
        angular_separation = df_coords.separation(proxima)
        near_proxima = angular_separation < 1 * u.deg  # Within 1 degree
        filtered_df = df[near_proxima]
        if filtered_df.empty:
            return "No neutrino events near Proxima Centauri."
        # FFT for temporal analysis
        times = filtered_df['time'].sort_values()
        if len(times) < 2:
            return "Insufficient data for temporal analysis."
        dt = np.diff(times)
        freq = scipy.fft.fftfreq(len(dt), d=np.mean(dt))
        fft_values = scipy.fft.fft(dt)
        power = np.abs(fft_values)**2
        # Simple scoring for artificial signals (e.g., high power in non-random frequencies)
        score = np.max(power) / np.mean(power) if np.mean(power) > 0 else 0
        results = {"events": len(filtered_df), "max_energy": filtered_df['energy'].max(), "score": score}
        save_json("neutrino_results.json", results)
        return f"Analyzed {len(filtered_df)} events near Proxima Centauri. Signal score: {score:.2f}. Results saved to neutrino_results.json."
    except Exception as e:
        log_error(f"Error analyzing neutrinos: {str(e)}")
        return f"Error analyzing neutrinos: {str(e)}"

def add_command(new_command):
    try:
        # Backup source
        shutil.copy(SOURCE_FILE, BACKUP_FILE)
        with open(SOURCE_FILE, "r") as f:
            lines = f.readlines()
        # Find and modify ALLOWED_ACTIONS
        for i, line in enumerate(lines):
            if "ALLOWED_ACTIONS = " in line:
                lines[i] = line.rstrip()[:-1] + f', "{new_command}"]\n"  # Append new command
                break
        with open(SOURCE_FILE, "w") as f:
            f.writelines(lines)
        # Restart script
        os.execv(sys.executable, [sys.executable] + sys.argv)
        return f"Added '{new_command}' to allowed actions and restarted script."
    except Exception as e:
        log_error(f"Error adding command: {str(e)}")
        return f"Error adding command: {str(e)}"

def optimize_web_search():
    global WEB_SNIPPET_LENGTH, SOURCE_PRIORITY
    WEB_SNIPPET_LENGTH = 500  # Increase length
    SOURCE_PRIORITY.append("agriculture")  # Add priority
    return "Optimized web search: increased snippet length to 500 and added 'agriculture' to source priority."

def track_command(command, success=True):
    try:
        history = load_command_history()
        feedback = load_feedback_history()
        history[command] = history.get(command, {"count": 0})
        history[command]["count"] += 1
        feedback[command] = feedback.get(command, {"success": 0, "total": 0})
        feedback[command]["success"] += 1 if success else 0
        feedback[command]["total"] += 1
        save_command_history(history)
        save_feedback_history(feedback)
    except Exception as e:
        log_error(f"Failed to track command: {str(e)}")

def track_context(command, response):
    try:
        context = load_context_history()
        context.append({"command": command, "response": response})
        save_context_history(context)
    except Exception as e:
        log_error(f"Failed to track context: {str(e)}")

def execute_command(command):
    if not command:
        return "No command received."
    
    words = command.split()
    intent = words[0].lower()
    command_lower = command.lower()
    
    if intent not in ALLOWED_ACTIONS:
        track_command(command, success=False)
        response = "Command not recognized. Try 'open chrome', 'type hello', 'weather', 'take picture', 'what do you see', 'suggest command', 'list plants', 'toggle suggestions', 'stop'."
        track_context(command, response)
        update_dialogue_context(command, response)
        return response

    track_command(command, success=True)

    if intent == "open" and len(words) > 1:
        app = words[1].lower()
        if app not in ALLOWED_APPS:
            track_command(command, success=False)
            response = f"Application '{app}' not allowed. Try {', '.join(ALLOWED_APPS)}."
            track_context(command, response)
            update_dialogue_context(command, response)
            return response
        try:
            if os.name == "nt":
                subprocess.run(["start", "", app], shell=True)
            response = f"Opening {app} with utmost efficiency, sir."
            track_context(command, response)
            update_dialogue_context(command, response)
            return response
        except Exception as e:
            track_command(command, success=False)
            response = f"Failed to open {app}: {str(e)}."
            track_context(command, response)
            update_dialogue_context(command, response)
            return response
    elif intent == "type" and len(words) > 1:
=======
        if not objects:
            img = cv2.imread(IMAGE_PATH)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_green = (40, 40, 40)
            upper_green = (80, 255, 255)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            green_pixels = cv2.countNonZero(mask_green)
            lower_brown = (10, 50, 50)
            upper_brown = (20, 255, 200)
            mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
            brown_pixels = cv2.countNonZero(mask_brown)
            total_pixels = img.shape[0] * img.shape[1]
            green_percentage = (green_pixels / total_pixels) * 100
            brown_percentage = (brown_pixels / total_pixels) * 100
            if green_percentage > 10 or brown_percentage > 10:
                return f"We couldn't identify specific objects, but the image contains {green_percentage:.2f}% green and {brown_percentage:.2f}% brown, which might indicate a gourd."
            return "We couldn't identify any objects in the image."
    except Exception as e:
        return f"Error recognizing objects: {str(e)}"

def scan_files():
    """Scan for files in the specified directory."""
    try:
        files = os.listdir(SCAN_DIRECTORY)
        if not files:
            return f"No files found in {SCAN_DIRECTORY}."
        file_list = ", ".join(files)
        return f"Here are the files in {SCAN_DIRECTORY}: {file_list}."
    except Exception as e:
        return f"Error scanning files: {str(e)}."

def read_file(filename):
    """Read the contents of a specified file."""
    file_path = os.path.join(SCAN_DIRECTORY, filename)
    if not os.path.exists(file_path):
        return f"File {filename} not found in {SCAN_DIRECTORY}."
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return f"File {filename} is empty."
        if len(content) > 500:
            content = content[:500] + "... (truncated)"
        return f"Contents of {filename}: {content}"
    except Exception as e:
        return f"Error reading file {filename}: {str(e)}."

def load_search_cache():
    """Load search cache from JSON file."""
    if os.path.exists(SEARCH_CACHE_FILE):
        with open(SEARCH_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_search_cache(cache):
    """Save search cache to JSON file."""
    with open(SEARCH_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

def load_reminders():
    """Load reminders from JSON file."""
    if os.path.exists(REMINDERS_FILE):
        with open(REMINDERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_reminders(reminders):
    """Save reminders to JSON file."""
    with open(REMINDERS_FILE, "w") as f:
        json.dump(reminders, f, indent=4)

def set_reminder(task):
    """Set a reminder for a task."""
    reminders = load_reminders()
    reminders[task] = "pending"
    save_reminders(reminders)
    return f"Reminder set for: {task}"

def list_reminders():
    """List all reminders."""
    reminders = load_reminders()
    if not reminders:
        return "No reminders set."
    reminder_list = ", ".join(reminders.keys())
    return f"Current reminders: {reminder_list}"

def view_code():
    """Read and display the contents of this script."""
    try:
        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            code = f.read()
        # Limit the output to avoid overwhelming the user
        if len(code) > 1000:
            code = code[:1000] + "... (truncated)"
        return f"Here is my code: {code}"
    except Exception as e:
        return f"Error reading my code: {str(e)}"

def add_command(command_name):
    """Add a new command to ALLOWED_ACTIONS and execute_command()."""
    try:
        # Create a backup of the current script
        shutil.copyfile(SOURCE_FILE, BACKUP_FILE)
        speak(f"Backup created at {BACKUP_FILE}")

        # Read the current script
        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Prepare new command logic
        command_name_clean = command_name.replace(" ", "_")  # Replace spaces with underscores for function name
        new_function = f"""
def {command_name_clean}():
    \"\"\"Execute the {command_name} command.\"\"\"
    if "{command_name}" == "local weather":
        return search_web("weather forecast Cleveland TN")
    return "This command is a placeholder. I need more details to implement its functionality."
"""
        new_lines = []
        modified_allowed = False
        modified_execute = False
        in_execute_command = False

        # Modify ALLOWED_ACTIONS and add the new command logic
        for line in lines:
            # Add to ALLOWED_ACTIONS
            if line.strip().startswith("ALLOWED_ACTIONS ="):
                new_lines.append(line.rstrip('\n')[:-1] + f', "{command_name}"]\n')
                modified_allowed = True
            # Check if we're in execute_command() to add the new command logic
            elif line.strip().startswith("def execute_command(command):"):
                in_execute_command = True
                new_lines.append(line)
            elif in_execute_command and line.strip().startswith("if first_word == \"set\" and len(words) > 1 and words[1].lower() == \"reminder\":"):
                new_lines.append(f'    if action == "{command_name}":\n')
                new_lines.append(f'        return {command_name_clean}()\n')
                new_lines.append(line)
                modified_execute = True
            else:
                new_lines.append(line)

        # Append the new function at the end of the file
        new_lines.append(new_function)

        if not modified_allowed or not modified_execute:
            return "Could not modify ALLOWED_ACTIONS or execute_command(). Please check my code structure."

        # Write the modified script
        with open(SOURCE_FILE, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        # Restart the script to apply changes
        speak(f"Added new command '{command_name}'. Restarting now...")
        subprocess.run([sys.executable, SOURCE_FILE])
        sys.exit(0)  # Exit the current process
    except Exception as e:
        return f"Error adding command: {str(e)}"

def optimize_web_search():
    """Optimize web search by adjusting parameters."""
    try:
        # Create a backup of the current script
        shutil.copyfile(SOURCE_FILE, BACKUP_FILE)
        speak(f"Backup created at {BACKUP_FILE}")

        # Read the current script
        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Modify web search settings (increase snippet length and add new source priority)
        new_snippet_length = WEB_SNIPPET_LENGTH + 100
        new_source_priority = WEB_SOURCE_PRIORITY + ["agriculture"]
        modified_snippet = False
        modified_priority = False
        new_lines = []
        for line in lines:
            if line.strip().startswith("WEB_SNIPPET_LENGTH ="):
                new_lines.append(f"WEB_SNIPPET_LENGTH = {new_snippet_length}\n")
                modified_snippet = True
            elif line.strip().startswith("WEB_SOURCE_PRIORITY ="):
                new_lines.append(f"WEB_SOURCE_PRIORITY = {json.dumps(new_source_priority)}\n")
                modified_priority = True
            else:
                new_lines.append(line)

        if not modified_snippet or not modified_priority:
            return "Could not find web search settings to modify."

        # Write the modified script
        with open(SOURCE_FILE, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        # Restart the script to apply changes
        speak(f"Optimized web search: snippet length increased to {new_snippet_length}, added 'agriculture' to source priority. Restarting now...")
        subprocess.run([sys.executable, SOURCE_FILE])
        sys.exit(0)  # Exit the current process
    except Exception as e:
        return f"Error optimizing web search: {str(e)}"

def diagnose_issues():
    """Diagnose common issues and suggest fixes."""
    try:
        issues = []
        # Check SerpAPI connectivity and quota
        api_key = "5b664dcd4b5349cfc97ffbea7dc2848d89a522c4c4dbed5bca319af71715e30b"
        test_url = f"https://serpapi.com/search?api_key={api_key}&q=test"
        response = requests.get(test_url)
        if response.status_code != 200:
            issues.append(f"SerpAPI issue: Status code {response.status_code}. Possible quota limit reached or invalid key. Check your SerpAPI account at serpapi.com.")
        else:
            data = response.json()
            if "error" in data:
                issues.append(f"SerpAPI error: {data['error']}. Check your API key or quota at serpapi.com.")

        # Check file permissions
        if not os.access(SOURCE_FILE, os.W_OK):
            issues.append(f"Cannot write to my source file {SOURCE_FILE}. Check file permissions.")

        if not issues:
            return "No issues detected. I seem to be running smoothly!"
        return "Detected issues:\n" + "\n".join(issues)
    except Exception as e:
        return f"Error diagnosing issues: {str(e)}"

def improve_voice_recognition():
    """Improve voice recognition by adjusting Vosk settings."""
    try:
        # Create a backup of the current script
        shutil.copyfile(SOURCE_FILE, BACKUP_FILE)
        speak(f"Backup created at {BACKUP_FILE}")

        # Read the current script
        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Modify the Vosk settings (increase beam and lattice-beam)
        new_beam = VOSK_BEAM + 5
        new_lattice_beam = VOSK_LATTICE_BEAM + 2
        modified = False
        new_lines = []
        for line in lines:
            if line.strip().startswith("VOSK_BEAM ="):
                new_lines.append(f"VOSK_BEAM = {new_beam}\n")
                modified = True
            elif line.strip().startswith("VOSK_LATTICE_BEAM ="):
                new_lines.append(f"VOSK_LATTICE_BEAM = {new_lattice_beam}\n")
                modified = True
            else:
                new_lines.append(line)

        if not modified:
            return "Could not find Vosk settings to modify."

        # Write the modified script
        with open(SOURCE_FILE, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        # Restart the script to apply changes
        speak(f"Improved voice recognition settings to beam={new_beam} and lattice-beam={new_lattice_beam}. Restarting now...")
        subprocess.run([sys.executable, SOURCE_FILE])
        sys.exit(0)  # Exit the current process
    except Exception as e:
        return f"Error improving voice recognition: {str(e)}"

def local_weather():
    """Execute the local weather command."""
    if "local weather" == "local weather":
        return search_web("weather forecast Cleveland TN")
    return "This command is a placeholder. I need more details to implement its functionality."

def search_web(query):
    """Search the web for information using SerpAPI."""
    cache = load_search_cache()
    if query in cache:
        return f"From cache: {cache[query]}"
    try:
        api_key = "5b664dcd4b5349cfc97ffbea7dc2848d89a522c4c4dbed5bca319af71715e30b"
        url = f"https://serpapi.com/search?api_key={api_key}&q={query}&location=Cleveland,TN"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "knowledge_graph" in data and "description" in data["knowledge_graph"]:
            description = data["knowledge_graph"]["description"]
            if len(description) > WEB_SNIPPET_LENGTH:
                description = description[:WEB_SNIPPET_LENGTH] + "..."
            cache[query] = description
            save_search_cache(cache)
            return f"Here's what I found: {description}"
        if "organic_results" in data and data["organic_results"]:
            for result in data["organic_results"]:
                snippet = result.get("snippet", "")
                link = result.get("link", "").lower()
                if snippet and any(priority in link for priority in WEB_SOURCE_PRIORITY):
                    if len(snippet) > WEB_SNIPPET_LENGTH:
                        snippet = snippet[:WEB_SNIPPET_LENGTH] + "..."
                    cache[query] = snippet
                    save_search_cache(cache)
                    return f"Here's what I found from a prioritized source: {snippet}"
            snippet = data["organic_results"][0].get("snippet", "")
            if snippet:
                if len(snippet) > WEB_SNIPPET_LENGTH:
                    snippet = snippet[:WEB_SNIPPET_LENGTH] + "..."
                cache[query] = snippet
                save_search_cache(cache)
                return f"Here's what I found: {snippet}"
        return "I couldn't find a clear answer on the web. Can you try rephrasing your query?"
    except Exception as e:
        return f"Error searching the web: {str(e)}. Please ensure you have an internet connection."

def execute_command(command):
    """Process and execute commands."""
    if not command:
        return "No command received."

    words = command.split()
    if not words:
        return "Invalid command."

    # Handle multi-word commands by joining words and comparing
    command_lower = command.lower()
    # Normalize by removing spaces for matching
    normalized_command = command_lower.replace(" ", "")
    action = command_lower

    # Extract the first word for commands like "open", "type", or "read file" that need additional args
    first_word = words[0].lower()

    # Map variations to allowed actions
    if "check cpu" in command_lower:
        action = "cpu"
    if "good night" in command_lower or normalized_command == "goodnight":
        action = "goodnight"
    if "scan files" in command_lower or normalized_command == "scanfiles":
        action = "scan files"
    if first_word == "read" and len(words) > 1 and words[1].lower() == "file":
        action = "read file"
    if first_word == "look" and len(words) > 1 and words[1].lower() == "up":
        action = "look up"
    if first_word == "gardening" and len(words) > 2 and words[1].lower() == "tips" and words[2].lower() == "for":
        action = "gardening tips for"
    if first_word == "set" and len(words) > 1 and words[1].lower() == "reminder":
        action = "set reminder"
    if action == "list reminders":
        action = "list reminders"
    if first_word == "view" and len(words) > 1 and words[1].lower() == "code":
        action = "view code"
    if first_word == "improve" and len(words) > 2 and words[1].lower() == "voice" and words[2].lower() == "recognition":
        action = "improve voice recognition"
    if first_word == "add" and len(words) > 1 and words[1].lower() == "command":
        action = "add command"
    if first_word == "optimize" and len(words) > 2 and words[1].lower() == "web" and words[2].lower() == "search":
        action = "optimize web search"
    if first_word == "diagnose" and len(words) > 1 and words[1].lower() == "issues":
        action = "diagnose issues"

    if action not in ALLOWED_ACTIONS and first_word not in ["open", "type", "read", "look", "gardening", "set", "view", "improve", "add", "optimize", "diagnose"]:
        # Fallback to web search for unknown commands
        return search_web(command)

    # Track the command (except for "suggest command")
    if action != "suggest command":
        track_command(command)

    if first_word == "open" and len(words) > 1:
        app = words[1].lower()
        if app not in ALLOWED_APPS:
            return f"Application '{app}' not allowed. Try {', '.join(ALLOWED_ACTIONS)}."
        try:
            if os.name == "nt":  # Windows
                subprocess.run(["start", "", app], shell=True)
            elif os.name == "posix" and os.uname().sysname == "Darwin":  # macOS
                subprocess.run(["open", "-a", app.capitalize()])
            else:  # Linux
                subprocess.run([app])
            return f"We have opened {app}."
        except Exception as e:
            return f"Failed to open {app}: {str(e)}."

    elif first_word == "type" and len(words) > 1:
>>>>>>> origin/main
        text = " ".join(words[1:])
        try:
            pyautogui.write(text, interval=0.1)
            pyautogui.press("enter")
<<<<<<< HEAD
            response = f"Typed '{text}' with precision."
            track_context(command, response)
            update_dialogue_context(command, response)
            return response
        except Exception as e:
            track_command(command, success=False)
            response = f"Failed to type: {str(e)}."
            track_context(command, response)
            update_dialogue_context(command, response)
            return response
    elif intent == "weather":
        response = local_weather()
        track_context(command, response)
        update_dialogue_context(command, response)
        suggestion = generate_proactive_suggestion()
        if suggestion:
            response += f"\n{suggestion}"
        return response
    elif intent == "take":
        if "picture" in command_lower:
            response = capture_image()
            track_context(command, response)
            update_dialogue_context(command, response)
            return response
    elif intent == "what":
        if "do you see" in command_lower:
            response = recognize_objects()
            track_context(command, response)
            update_dialogue_context(command, response)
            return response
    elif intent == "list":
        if "plants" in command_lower:
            response = list_plants()
            track_context(command, response)
            update_dialogue_context(command, response)
            suggestion = generate_proactive_suggestion()
            if suggestion:
                response += f"\n{suggestion}"
            return response
    elif intent == "toggle":
        if "learning" in command_lower:
            response = toggle_learning()
        elif "suggestions" in command_lower:
            response = toggle_suggestions()
        track_context(command, response)
        update_dialogue_context(command, response)
        return response
    elif intent == "cpu":
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            response = f"CPU usage: {cpu_usage}% | Memory usage: {memory_usage}%"
        except Exception as e:
            response = f"Error checking system usage: {str(e)}"
        track_context(command, response)
        update_dialogue_context(command, response)
        return response
    elif intent == "analyze":
        if "neutrinos" in command_lower:
            response = analyze_neutrinos()
            track_context(command, response)
            update_dialogue_context(command, response)
            return response
    elif intent == "add":
        if "command" in command_lower and len(words) > 2:
            new_command = " ".join(words[2:])
            response = add_command(new_command)
            track_context(command, response)
            update_dialogue_context(command, response)
            return response
    elif intent == "optimize":
        if "web search" in command_lower:
            response = optimize_web_search()
            track_context(command, response)
            update_dialogue_context(command, response)
            return response
    elif intent == "stop":
        response = "Shutting down. Goodbye."
        track_context(command, response)
        update_dialogue_context(command, response)
        return response

    response = "Command not fully understood. Try a more specific request."
    track_command(command, success=False)
    track_context(command, response)
    update_dialogue_context(command, response)
    return response

def create_gui():
    global gui_output
    root = tk.Tk()
    root.title("CORVUS - Your Homestead Companion")
    root.geometry("600x400")
    root.attributes('-topmost', True)
    root.update()
    root.attributes('-topmost', False)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 600
    window_height = 400
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, expand=True)
    plant_frame = ttk.Frame(notebook)
    notebook.add(plant_frame, text="Plant Identification")
    ttk.Button(plant_frame, text="Take Picture", command=lambda: speak(capture_image())).pack(pady=5)
    ttk.Button(plant_frame, text="What Do You See?", command=lambda: speak(recognize_objects())).pack(pady=5)
    ttk.Button(plant_frame, text="Toggle Learning", command=lambda: speak(toggle_learning())).pack(pady=5)
    ttk.Button(plant_frame, text="Toggle Suggestions", command=lambda: speak(toggle_suggestions())).pack(pady=5)
    command_frame = ttk.Frame(notebook)
    notebook.add(command_frame, text="Commands")
    ttk.Label(command_frame, text="Enter Command:").pack(pady=5)
    command_entry = ttk.Entry(command_frame, width=50)
    command_entry.pack(pady=5)
    gui_output = scrolledtext.ScrolledText(command_frame, width=60, height=10, wrap=tk.WORD)
    gui_output.pack(pady=5)
    ttk.Button(command_frame, text="Execute", command=lambda: speak(execute_command(command_entry.get()))).pack(pady=5)
    flask_thread = threading.Thread(target=lambda: app.run(debug=False, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()
    learning_thread = threading.Thread(target=proactive_learning_thread)
    learning_thread.daemon = True
    learning_thread.start()
    voice_thread = threading.Thread(target=listen_for_activation)
    voice_thread.daemon = True
    voice_thread.start()
    speak("CORVUS online. How may I assist you today, sir?")
    root.mainloop()

if __name__ == "__main__":
    create_gui()
=======
            return f"We have typed: {text}"
        except Exception as e:
            return f"Failed to type: {str(e)}."

    elif action == "cpu":
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        return f"System status: CPU usage at {cpu_percent}%, memory usage at {memory.percent}%."

    elif action == "weather":
        try:
            api_key = "YOUR_API_KEY"  # Replace with OpenWeatherMap key
            city = "YOUR_CITY"  # e.g., "Cleveland,TN"
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url).json()
            if response.get("weather"):
                weather = response["weather"][0]["description"]
                temp = response["main"]["temp"]
                return f"Homestead weather: {weather}, {temp}°C. Optimal for current tasks."
            return "We could not retrieve weather data."
        except Exception as e:
            return f"Weather fetch failed: {str(e)}."

    elif action == "take picture":
        return capture_image()

    elif action == "what do you see":
        return recognize_objects()

    elif action == "suggest command":
        return suggest_command()

    elif action == "goodnight":
        return "Goodnight! Sleep well, and may your dreams be filled with the wonders of your homestead."

    elif action == "scan files":
        return scan_files()

    elif action == "read file" and len(words) > 2:
        filename = " ".join(words[2:])
        return read_file(filename)

    elif action == "look up" and len(words) > 2:
        query = " ".join(words[2:])
        return search_web(query)

    elif action == "gardening tips for" and len(words) > 3:
        query = " ".join(words)
        return search_web(query)

    elif action == "set reminder" and len(words) > 2:
        task = " ".join(words[2:])
        return set_reminder(task)

    elif action == "list reminders":
        return list_reminders()

    elif action == "view code":
        return view_code()

    elif action == "improve voice recognition":
        return improve_voice_recognition()

    elif action == "add command" and len(words) > 2:
        command_name = " ".join(words[2:])
        return add_command(command_name)

    elif action == "optimize web search":
        return optimize_web_search()

    elif action == "diagnose issues":
        return diagnose_issues()

    elif action == "local weather":
        return local_weather()

    elif action == "analyze neutrinos":
        return analyze_neutrinos()

    elif action == "stop":
        return "Shutting down. Goodbye."

    return "Command not recognized. Try 'open chrome', 'type hello', 'cpu', 'weather', 'take picture', 'what do you see', 'suggest command', 'goodnight', 'scan files', 'read file', 'look up', 'gardening tips for', 'set reminder', 'list reminders', 'view code', 'improve voice recognition', 'add command', 'optimize web search', 'diagnose issues', 'local weather', 'analyze neutrinos', or 'stop'."

# Flask route for phone commands
@app.route("/command", methods=["POST"])
def handle_command():
    data = request.json
    command = data.get("command", "").lower()
    if not command:
        return jsonify({"response": "No command provided."}), 400
    response = execute_command(command)
    speak(response)
    return jsonify({"response": response})

def run_flask():
    """Run Flask server."""
    app.run(host="0.0.0.0", port=5000)

def main():
    """Main function."""
    speak("Initializing CORVUS for computer and homestead control. Ready for commands.")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    while True:
        command = listen_local()
        if command:
            response = execute_command(command)
            speak(response)
            if "stop" in command:
                break
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        speak("System offline. Until next time!")
>>>>>>> origin/main
