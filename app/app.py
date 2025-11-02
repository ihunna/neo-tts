"""
Neo TTS Flask Application
Modern modular TTS system with Kokoro
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import time
import json
import csv
from pathlib import Path
import importlib
import logging
from device_utils import (
    get_device_info, monitor_gpu_usage, PerformanceTimer,
    log_model_device_info, setup_logging
)

# Add current directory and project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)



# Initialize Flask app
app = Flask(__name__)

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "output")
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Model registry
MODELS = {
    "kokoro": {
        "name": "Kokoro",
        "description": "Fast lightweight TTS for local mac",
        "module": "models.kokoro"
    }
}

# Cache for loaded model modules
_model_modules = {}

def get_model_module(model_name):
    """Load model module dynamically."""
    if model_name not in _model_modules:
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        module_name = MODELS[model_name]["module"]
        try:
            _model_modules[model_name] = importlib.import_module(module_name)
        except ImportError as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    return _model_modules[model_name]

def log_generation(model, speaker, text, output_path):
    """Log generation to CSV."""
    log_file = os.path.join(LOGS_DIR, "results.csv")
    file_exists = os.path.exists(log_file)

    duration = 0.0
    try:
        import soundfile as sf
        audio, sr = sf.read(output_path)
        duration = len(audio) / sr
    except:
        pass

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "speaker", "text", "duration", "output_path"])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            model,
            speaker or "default",
            text[:100] + "..." if len(text) > 100 else text,
            f"{duration:.2f}",
            output_path
        ])

@app.route("/")
def index():
    """Render main page."""
    return render_template("index.html", models=MODELS)

@app.route("/api/voices/<model_name>")
def get_voices(model_name):
    """Get available voices for a model."""
    try:
        module = get_model_module(model_name)
        voices = module.list_voices()
        return jsonify({"voices": voices})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/device-info")
def get_device_info_endpoint():
    """Get comprehensive device information."""
    try:
        device_info = get_device_info()
        gpu_usage = monitor_gpu_usage()
        return jsonify({
            "device_info": device_info,
            "gpu_usage": gpu_usage,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/generate", methods=["POST"])
def generate_audio():
    """Generate audio using specified model."""
    try:
        data = request.json
        model_name = data.get("model")
        voice = data.get("voice")
        text = data.get("text", "").strip()

        if not model_name or model_name not in MODELS:
            return jsonify({"error": "Invalid model"}), 400

        if not text:
            return jsonify({"error": "Text is required"}), 400

        # Load model module
        module = get_model_module(model_name)

        # Generate unique filename
        timestamp = int(time.time())
        filename = f"{model_name}_{timestamp}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Performance timing and device monitoring
        generation_start = time.time()
        gpu_usage_before = monitor_gpu_usage()

        try:
            # Generate audio
            with PerformanceTimer(f"{model_name} generation ({voice or 'default'})"):
                result_path = module.generate_audio(text, voice, output_path)

            generation_time = time.time() - generation_start
            gpu_usage_after = monitor_gpu_usage()

            # Calculate audio duration
            audio_duration = 0.0
            try:
                import soundfile as sf
                audio, sr = sf.read(result_path)
                audio_duration = len(audio) / sr
            except:
                pass

            # Log the generation with performance info
            log_generation(model_name, voice, text, result_path)

            # Return relative path for web access
            web_path = f"/static/output/{filename}"
            return jsonify({
                "audio_url": web_path,
                "model": model_name,
                "voice": voice,
                "audio_duration": round(audio_duration, 2),
                "generation_time": round(generation_time, 2),
                "gpu_usage": gpu_usage_after
            })

        except Exception as e:
            generation_time = time.time() - generation_start
            app.logger.error(".2f")
            raise e

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/static/output/<path:filename>")
def serve_audio(filename):
    """Serve generated audio files."""
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    print("🚀 Neo TTS Server")
    print("Available models:", list(MODELS.keys()))
    print("Visit http://localhost:5000")
    app.run(debug=True, port=5000)
