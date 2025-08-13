 # CORVUS AI Assistant

An AI assistant for computer and homestead control built in Python.

## Overview
CORVUS is a Python-based AI assistant designed to manage and automate tasks for both computer control and homestead activities. This combined version integrates advanced voice recognition, GUI, plant identification from the local version with neutrino analysis, self-modification, and system monitoring from the original design.

## Purpose
- Automate routine computer tasks (e.g., opening applications, typing text).
- Provide homestead-specific functionalities like weather updates, plant identification, and gardening tips.
- Perform advanced scientific analysis, such as detecting potential artificial neutrino signals near Proxima Centauri.
- Demonstrate modularity, error handling, and self-modification capabilities in Python.

## Key Features
- **Command Processing**: Supports commands like `open`, `type`, `cpu`, `weather`, `take picture`, `analyze neutrinos`, and more. Uses an allowlist for safety.
- **Voice Recognition**: Integrates `pvporcupine` for wake-word and `pvrhino` for intent, with `whisper` fallback.
- **Web Search and Caching**: Uses SerpAPI with caching for efficiency.
- **Computer Vision**: Utilizes YOLOv8 for object detection and PlantNet API for plant identification.
- **System Monitoring**: Monitors CPU and memory with `psutil`.
- **Neutrino Data Analysis**: Analyzes IceCube data for artificial signals using `pandas`, `numpy`, `scipy`, `astropy`.
- **Self-Modification**: Adds new commands dynamically with backups.
- **GUI**: `tkinter`-based interface for commands and plant identification.
- **Proactive Suggestions**: Context-aware suggestions for homestead tasks.

## Dependencies
- Python 3.8+
- Libraries: See `requirements.txt` (includes pyautogui, psutil, flask, opencv-python, ultralytics, whisper, pvporcupine, pvrhino, langchain, pandas, scipy, astropy, etc.)
- External Services: SerpAPI, PlantNet API, Picovoice (keys required).

## Setup Instructions
1. Clone the repository: `git clone https://github.com/FrankleFry1/corvus-ai-assistant.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure paths and API keys in `corvus_control.py`.
4. Run: `python corvus_control.py`

## Usage Examples
- `open notepad`: Opens Notepad.
- `cpu`: Displays CPU/memory usage.
- `analyze neutrinos`: Analyzes neutrino data.
- `add command check battery`: Adds a new command and restarts.

## License
MIT License