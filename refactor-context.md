You are an expert macOS + Python AI engineer.

I want you to completely remove everything related to the old TTS models — Coqui XTTS, Bark, and OpenVoice — from my Flask-based local TTS project.  
That includes all:
- Old dependencies (TTS, Bark, OpenVoice)
- Any references in setup scripts, requirements, or code
- Unused virtual environments and model loaders

Then rebuild the entire project **from scratch** using a modern modular architecture that supports **multi-speaker voice selection**.

---

## 🧱 1️⃣ Environment Setup

- Create a fresh `venv` using **Python 3.10** (compatible with all target models).
- Detect mac architecture (Apple Silicon `arm64` vs Intel `x86_64`).
- Install cleanly:
  - `torch` (Metal backend for Apple Silicon or CPU for Intel)
  - `flask`, `soundfile`, `accelerate`, `numpy`, `requests`
  - Any model-specific dependencies for Kokoro, Zonos, Chatterbox, DIA, Orpheus, and CSM
- Automatically download or install the models from their sources (HuggingFace / GitHub)
- Save a `requirements-neo.txt` snapshot
- File: `setup-neo-tts.sh`

---

## ⚙️ 2️⃣ Flask App (app/app.py)

Rebuild the Flask app to:
- Load models dynamically (only one active at a time)
- Detect available **voices/speakers** per model (multi-speaker models)
- Provide the following functionality:
  - Dropdown to select model
  - Dropdown to select voice (auto-populated once model is selected)
  - Text input for the user
  - “Generate” button to synthesize speech
  - Auto-play or download the resulting audio file

All synthesis should happen via **POST** requests.

Save generated audio to `static/output.wav` and display a player after generation.

Each synthesis should be logged in `logs/results.csv` with:
timestamp, model, speaker, text, duration, output_path

yaml
Copy code

---

## 🧩 3️⃣ Modular Model Loader Architecture

Create a `models/` folder with one file per model:
models/
├── kokoro.py
├── chatterbox.py
├── zonos.py
├── dia.py
├── orpheus.py
├── csm.py

python
Copy code

Each file must expose a consistent interface:
```python
def list_voices() -> list:
    """Return a list of available speakers/voices for this model."""

def generate_audio(text: str, voice: str = None, output_path: str = "static/output.wav") -> str:
    """Generate audio for given text and optional voice, returning the saved file path."""
🗣️ 4️⃣ Multi-Speaker Voice Selection Logic
Kokoro: use available voice presets (male, female, etc.)

Chatterbox: list internal trained voices (e.g., p225, p226, etc.)

Zonos: allow the user to upload or specify a reference voice file for cloning (speaker_wav)

DIA, Orpheus, CSM: if single-speaker, disable the voice dropdown dynamically

When the user selects a model, fetch available voices from /get_voices?model=xxx endpoint and populate the dropdown.

💻 5️⃣ Frontend (templates/index.html)
Build a clean, minimal web UI (Tailwind or basic CSS):

Model dropdown

Voice dropdown (hidden/disabled for single-speaker models)

Text input area

Generate button

Embedded audio player (auto-updates after generation)

Optional small log display (show last generated model/voice)

Example layout:

less
Copy code
Select Model: [ Kokoro ▼ ]
Select Voice: [ voice_1 ▼ ]
Enter Text: [ "Hello world..." ]
[Generate 🔊]
[ Player ⏯️ ]
Use JavaScript to:

Dynamically fetch voices when model changes

Send POST requests for synthesis

Update the player without reloading the page

🧠 6️⃣ Model Management Details
Kokoro: Fast lightweight TTS for local mac (supports multiple voices)

Chatterbox: Conversational expressive model with multiple speakers

Zonos: Zero-shot voice cloning (optional user-uploaded voice file)

DIA: High-quality expressive model for narration

Orpheus: Singing / creative synthesis (optional mode)

CSM: Optional contextual module enhancing voice consistency

All models should be fully offline — no API calls.

🧾 7️⃣ Verification
At the end of the setup script:

Test-generate short audio for each model (e.g. “Hello from [model name]”)

Print confirmation for each model:

Copy code
✅ Kokoro OK
✅ Chatterbox OK
✅ Zonos OK
✅ DIA OK
✅ Orpheus OK
✅ CSM OK
🧰 8️⃣ Deliverables
Final project structure:

project-root/
│
├── setup-neo-tts.sh                # One-click setup script (Python 3.10 + models)
├── venv/                           # Virtual environment (auto-created)
│
├── app/
│   ├── app.py                      # Main Flask server
│   │
│   ├── templates/
│   │   └── index.html              # Frontend UI (model + voice selector)
│   │
│   ├── static/                     # Flask static directory
│   │   ├── output/                 # Generated audio files
│   │   │   └── output.wav
│   │   ├── css/                    # (Optional) Tailwind or custom styles
│   │   └── js/                     # (Optional) Frontend JS for dynamic updates
│   │
│   └── __init__.py                 # (Optional) Makes `app/` importable
│
├── models/
│   ├── kokoro.py                   # Base lightweight TTS model
│   ├── chatterbox.py               # Conversational expressive model
│   ├── zonos.py                    # Zero-shot voice cloning
│   ├── dia.py                      # Emotional intonation model
│   ├── orpheus.py                  # Singing/creative synthesis
│   └── csm.py                      # Contextual speaker model (enhancer)
│
├── logs/
│   └── results.csv                 # Logs: timestamp, model, voice, text, duration
│
└── requirements-neo.txt            # Frozen environment for reproducibility

🎯 End Goal
After setup, I should be able to:

Run bash setup-neo-tts.sh

Activate the virtualenv: source venv/bin/activate

Start the Flask server: python app/app.py

Visit http://localhost:5000

Select a model, choose a voice, type text, and play the generated output.

All generation must be local, modular, and logged.

🔥 Notes
Use modern Python best practices.

Keep each model loader modular and clean.

Ensure voice lists are dynamically fetched and cached.

Avoid large runtime dependencies that break macOS portability.