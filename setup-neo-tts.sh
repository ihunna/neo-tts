#!/usr/bin/env bash
set -e

echo "🚀 Neo TTS Setup – Modern Modular Architecture"

# -----------------------------------------------------
# 1️⃣ Ensure Homebrew & dependencies
# -----------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  echo "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "🧩 Installing ffmpeg + pkg-config..."
brew install ffmpeg pkg-config python@3.10 || true

# -----------------------------------------------------
# 2️⃣ Detect Mac architecture
# -----------------------------------------------------
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
  echo "✅ Detected Apple Silicon (arm64)"
  TORCH_EXTRA=""
else
  echo "✅ Detected Intel (x86_64)"
  TORCH_EXTRA=""
fi

# -----------------------------------------------------
# 3️⃣ Setup Python 3.10 virtual environment
# -----------------------------------------------------
cd "$(dirname "$0")"
PYTHON_PATH="$(brew --prefix python@3.10)/bin/python3.10"

echo "🐍 Creating virtual environment with Python $($PYTHON_PATH --version)"
$PYTHON_PATH -m venv venv
source venv/bin/activate

echo "🔄 Upgrading pip & tools..."
pip install --upgrade pip setuptools wheel

# -----------------------------------------------------
# 4️⃣ Install core dependencies
# -----------------------------------------------------
echo "🔥 Installing PyTorch (Metal/CPU build)..."
pip install torch==2.4.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

echo "📦 Installing core packages..."
pip install flask soundfile accelerate numpy requests

# -----------------------------------------------------
# 5️⃣ Install model-specific dependencies
# -----------------------------------------------------

# Kokoro dependencies
echo "🎵 Installing Kokoro dependencies..."
pip install scipy phonemizer kokoro

# -----------------------------------------------------
# 6️⃣ Download/install models
# -----------------------------------------------------
echo "📥 Downloading models..."

# Create models cache directory
mkdir -p ~/.cache/neo-tts

# Kokoro model
echo "📦 Downloading Kokoro model..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='hexgrad/Kokoro-82M', local_dir='models/kokoro_cache')
"

# -----------------------------------------------------
# 7️⃣ Create required directories
# -----------------------------------------------------
echo "📁 Creating project directories..."
mkdir -p app/templates
mkdir -p app/static/output
mkdir -p app/static/css
mkdir -p app/static/js
mkdir -p models
mkdir -p logs

# Create __init__.py
touch app/__init__.py

# -----------------------------------------------------
# 8️⃣ Verification tests
# -----------------------------------------------------
echo "🔍 Running verification tests..."

# Test Kokoro
echo "Testing Kokoro..."
python -c "
try:
    import sys
    sys.path.append('.')
    from models.kokoro import list_voices, generate_audio
    voices = list_voices()
    if voices:
        generate_audio('Hello from Kokoro', voices[0])
        print('✅ Kokoro OK')
    else:
        print('❌ Kokoro: No voices found')
except Exception as e:
    print(f'❌ Kokoro failed: {e}')
"

# -----------------------------------------------------
# 9️⃣ Freeze environment
# -----------------------------------------------------
pip freeze > requirements-neo.txt
echo
echo "✅ Environment setup complete!"
echo "To start developing:"
echo "1. source venv/bin/activate"
echo "2. python app/app.py (or flask run)"

echo "🎯 Ready to generate speech with Kokoro TTS!"
