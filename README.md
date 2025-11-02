# Neo TTS

A modern, modular Text-to-Speech (TTS) system built with Flask, featuring local inference with Kokoro TTS models.

## 🚀 Features

- **Modular Architecture**: Easily extensible with new TTS models
- **Web Interface**: Clean, responsive Flask-based UI
- **Local Inference**: Run completely offline with local models
- **Performance Monitoring**: Real-time GPU/CPU usage tracking
- **Multiple Voices**: Support for various voices and speakers
- **Audio Generation**: High-quality WAV output
- **Logging**: Comprehensive generation logs with performance metrics

## 🛠️ Supported Models

- **Kokoro**: Fast, lightweight TTS optimized for local macOS deployment

## 📋 Prerequisites

- macOS (Intel or Apple Silicon)
- Python 3.10+
- Homebrew

## 🏗️ Installation

### Automated Setup (Recommended)

Run the setup script to automatically configure your environment:

```bash
./setup-neo-tts.sh
```

This will:
- Install Homebrew dependencies (ffmpeg, pkg-config, Python 3.10)
- Create a virtual environment
- Install all required Python packages
- Download TTS models
- Set up project directories

### Manual Setup

If you prefer manual installation:

1. **Install system dependencies:**
   ```bash
   brew install ffmpeg pkg-config python@3.10
   ```

2. **Create virtual environment:**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements-neo.txt
   ```

4. **Download models:**
   ```bash
   python -c "
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id='hexgrad/Kokoro-82M', local_dir='models/kokoro_cache')
   "
   ```

## 🚀 Usage

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Start the server:**
   ```bash
   python app/app.py
   ```

3. **Open your browser:**
   Visit `http://localhost:5000`

4. **Generate speech:**
   - Select a TTS model
   - Choose a voice
   - Enter your text
   - Click generate

## 📁 Project Structure

```
local-tts-devlopment/
├── app/                    # Flask application
│   ├── static/            # Static assets (CSS, JS, output files)
│   ├── templates/         # HTML templates
│   ├── app.py            # Main Flask application
│   ├── device_utils.py   # Device monitoring utilities
│   └── __init__.py
├── models/                # TTS model implementations
│   └── kokoro.py         # Kokoro TTS wrapper
├── logs/                  # Generation logs and metrics
├── venv/                  # Virtual environment (created by setup)
├── setup-neo-tts.sh      # Automated setup script
├── requirements-neo.txt  # Python dependencies
└── README.md             # This file
```

## 🔧 Configuration

The application automatically detects your hardware and optimizes accordingly:

- **Apple Silicon (M1/M2/M3)**: Uses Metal acceleration when available
- **Intel Macs**: Falls back to CPU inference
- **GPU Monitoring**: Tracks VRAM usage and performance metrics

## 📊 Monitoring

The application provides real-time monitoring of:

- CPU/GPU usage during generation
- Generation time and audio duration
- Model performance metrics
- Device information

Access monitoring data via the `/api/device-info` endpoint.

## 🧪 Testing

Run the included verification tests:

```bash
# Test Kokoro model
python -c "
from models.kokoro import list_voices, generate_audio
voices = list_voices()
if voices:
    generate_audio('Hello from Kokoro', voices[0])
    print('✅ Kokoro OK')
"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Adding New Models

To add support for new TTS models:

1. Create a new module in `models/`
2. Implement the required interface:
   - `list_voices()`: Return available voices
   - `generate_audio(text, voice, output_path)`: Generate audio file
3. Register the model in `app/app.py` MODELS dictionary

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Kokoro TTS](https://github.com/hexgrad/kokoro) - Fast local TTS
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [PyTorch](https://pytorch.org/) - Machine learning framework

## 🐛 Issues

If you encounter any issues:

1. Check the logs in `app/logs/results.csv`
2. Ensure your virtual environment is activated
3. Verify all dependencies are installed
4. Check device compatibility

For bugs or feature requests, please open an issue on GitHub.
