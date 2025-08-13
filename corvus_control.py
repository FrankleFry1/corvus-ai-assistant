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

# Load YOLO model
model = YOLO("yolov8m.pt")

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
        text = " ".join(words[1:])
        try:
            pyautogui.write(text, interval=0.1)
            pyautogui.press("enter")
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