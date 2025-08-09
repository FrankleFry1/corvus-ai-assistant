import speech_recognition as sr
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

# Load YOLO model
model = YOLO("yolov8m.pt")

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
        text = " ".join(words[1:])
        try:
            pyautogui.write(text, interval=0.1)
            pyautogui.press("enter")
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