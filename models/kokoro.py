"""
Kokoro TTS Model Loader
Fast, lightweight TTS model for local macOS
Supports multiple voice presets (male/female)
"""

import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import sys
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.device_utils import get_optimal_device, log_model_device_info

# Suppress PyTorch warnings from Kokoro model internals
warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm.*deprecated.*", category=FutureWarning)

# Kokoro model cache
MODEL_CACHE = Path("models/kokoro_cache")
VOICES_DIR = MODEL_CACHE / "voices"

# Global model instance
_kokoro_pipeline = None

def _load_kokoro_model():
    """Load Kokoro model and voices lazily."""
    global _kokoro_pipeline

    if _kokoro_pipeline is not None:
        return _kokoro_pipeline

    try:
        # Import Kokoro (assuming it's installed)
        from kokoro import KPipeline

        # Create pipeline (automatically loads model)
        pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')  # American English

        # Load available voices from cache
        if VOICES_DIR.exists():
            for voice_file in VOICES_DIR.glob("*.pt"):
                voice_name = voice_file.stem  # e.g., 'af_alloy', 'am_adam', etc.
                try:
                    voice_tensor = torch.load(voice_file, weights_only=True)
                    pipeline.voices[voice_name] = voice_tensor
                except Exception as e:
                    print(f"Warning: Could not load voice {voice_name}: {e}")

        _kokoro_pipeline = pipeline
        return pipeline

    except ImportError:
        raise ImportError("Kokoro not installed. Run setup-neo-tts.sh first")
    except Exception as e:
        raise RuntimeError(f"Failed to load Kokoro model: {e}")

def _get_voice_mapping() -> dict:
    """Return mapping from formatted display names to raw voice codes."""
    return {
        # American English
        '🇺🇸 Alloy 🚺 (C)': 'af_alloy',
        '🇺🇸 Aoede 🚺 (C+)': 'af_aoede',
        '🇺🇸 Bella 🚺🔥 (A-)': 'af_bella',
        '🇺🇸 Heart 🚺❤️ (A)': 'af_heart',
        '🇺🇸 Jessica 🚺 (D)': 'af_jessica',
        '🇺🇸 Kore 🚺 (C+)': 'af_kore',
        '🇺🇸 Nicole 🚺🎧 (B-)': 'af_nicole',
        '🇺🇸 Nova 🚺 (C)': 'af_nova',
        '🇺🇸 River 🚺 (D)': 'af_river',
        '🇺🇸 Sarah 🚺 (C+)': 'af_sarah',
        '🇺🇸 Sky 🚺 (C-)': 'af_sky',

        '🇺🇸 Adam 🚹 (F+)': 'am_adam',
        '🇺🇸 Echo 🚹 (D)': 'am_echo',
        '🇺🇸 Eric 🚹 (D)': 'am_eric',
        '🇺🇸 Fenrir 🚹 (C+)': 'am_fenrir',
        '🇺🇸 Liam 🚹 (D)': 'am_liam',
        '🇺🇸 Michael 🚹 (C+)': 'am_michael',
        '🇺🇸 Onyx 🚹 (D)': 'am_onyx',
        '🇺🇸 Puck 🚹 (C+)': 'am_puck',
        '🇺🇸 Santa 🚹 (D-)': 'am_santa',

        # British English
        '🇬🇧 Alice 🚺 (D)': 'bf_alice',
        '🇬🇧 Emma 🚺 (B-)': 'bf_emma',
        '🇬🇧 Isabella 🚺 (C)': 'bf_isabella',
        '🇬🇧 Lily 🚺 (D)': 'bf_lily',

        '🇬🇧 Daniel 🚹 (D)': 'bm_daniel',
        '🇬🇧 Fable 🚹 (C)': 'bm_fable',
        '🇬🇧 George 🚹 (C)': 'bm_george',
        '🇬🇧 Lewis 🚹 (D+)': 'bm_lewis',

        # Japanese
        '🇯🇵 Alpha 🚺 (C+)': 'jf_alpha',
        '🇯🇵 Gongitsune 🚺 (C)': 'jf_gongitsune',
        '🇯🇵 Nezumi 🚺 (C-)': 'jf_nezumi',
        '🇯🇵 Tebukuro 🚺 (C)': 'jf_tebukuro',
        '🇯🇵 Kumo 🚹 (C-)': 'jm_kumo',

        # Mandarin Chinese
        '🇨🇳 Xiaobei 🚺 (D)': 'zf_xiaobei',
        '🇨🇳 Xiaoni 🚺 (D)': 'zf_xiaoni',
        '🇨🇳 Xiaoxiao 🚺 (D)': 'zf_xiaoxiao',
        '🇨🇳 Xiaoyi 🚺 (D)': 'zf_xiaoyi',
        '🇨🇳 Yunjian 🚹 (D)': 'zm_yunjian',
        '🇨🇳 Yunxi 🚹 (D)': 'zm_yunxi',
        '🇨🇳 Yunxia 🚹 (D)': 'zm_yunxia',
        '🇨🇳 Yunyang 🚹 (D)': 'zm_yunyang',

        # Spanish
        '🇪🇸 Dora 🚺': 'ef_dora',
        '🇪🇸 Alex 🚹': 'em_alex',
        '🇪🇸 Santa 🚹': 'em_santa',

        # French
        '🇫🇷 Siwis 🚺 (B-)': 'ff_siwis',

        # Hindi
        '🇮🇳 Alpha 🚺 (C)': 'hf_alpha',
        '🇮🇳 Beta 🚺 (C)': 'hf_beta',
        '🇮🇳 Omega 🚹 (C)': 'hm_omega',
        '🇮🇳 Psi 🚹 (C)': 'hm_psi',

        # Italian
        '🇮🇹 Sara 🚺 (C)': 'if_sara',
        '🇮🇹 Nicola 🚹 (C)': 'im_nicola',

        # Brazilian Portuguese
        '🇧🇷 Dora 🚺': 'pf_dora',
        '🇧🇷 Alex 🚹': 'pm_alex',
        '🇧🇷 Santa 🚹': 'pm_santa',
    }

def list_voices() -> list:
    """
    Return a list of available speakers/voices for Kokoro.
    Returns formatted voice names with flags, genders, and quality grades.
    """
    try:
        pipeline = _load_kokoro_model()
        raw_voices = list(pipeline.voices.keys())

        # Voice formatting mapping
        voice_formats = {
            # American English
            'af_alloy': '🇺🇸 Alloy 🚺 (C)',
            'af_aoede': '🇺🇸 Aoede 🚺 (C+)',
            'af_bella': '🇺🇸 Bella 🚺🔥 (A-)',
            'af_heart': '🇺🇸 Heart 🚺❤️ (A)',
            'af_jessica': '🇺🇸 Jessica 🚺 (D)',
            'af_kore': '🇺🇸 Kore 🚺 (C+)',
            'af_nicole': '🇺🇸 Nicole 🚺🎧 (B-)',
            'af_nova': '🇺🇸 Nova 🚺 (C)',
            'af_river': '🇺🇸 River 🚺 (D)',
            'af_sarah': '🇺🇸 Sarah 🚺 (C+)',
            'af_sky': '🇺🇸 Sky 🚺 (C-)',

            'am_adam': '🇺🇸 Adam 🚹 (F+)',
            'am_echo': '🇺🇸 Echo 🚹 (D)',
            'am_eric': '🇺🇸 Eric 🚹 (D)',
            'am_fenrir': '🇺🇸 Fenrir 🚹 (C+)',
            'am_liam': '🇺🇸 Liam 🚹 (D)',
            'am_michael': '🇺🇸 Michael 🚹 (C+)',
            'am_onyx': '🇺🇸 Onyx 🚹 (D)',
            'am_puck': '🇺🇸 Puck 🚹 (C+)',
            'am_santa': '🇺🇸 Santa 🚹 (D-)',

            # British English
            'bf_alice': '🇬🇧 Alice 🚺 (D)',
            'bf_emma': '🇬🇧 Emma 🚺 (B-)',
            'bf_isabella': '🇬🇧 Isabella 🚺 (C)',
            'bf_lily': '🇬🇧 Lily 🚺 (D)',

            'bm_daniel': '🇬🇧 Daniel 🚹 (D)',
            'bm_fable': '🇬🇧 Fable 🚹 (C)',
            'bm_george': '🇬🇧 George 🚹 (C)',
            'bm_lewis': '🇬🇧 Lewis 🚹 (D+)',

            # Japanese
            'jf_alpha': '🇯🇵 Alpha 🚺 (C+)',
            'jf_gongitsune': '🇯🇵 Gongitsune 🚺 (C)',
            'jf_nezumi': '🇯🇵 Nezumi 🚺 (C-)',
            'jf_tebukuro': '🇯🇵 Tebukuro 🚺 (C)',
            'jm_kumo': '🇯🇵 Kumo 🚹 (C-)',

            # Mandarin Chinese
            'zf_xiaobei': '🇨🇳 Xiaobei 🚺 (D)',
            'zf_xiaoni': '🇨🇳 Xiaoni 🚺 (D)',
            'zf_xiaoxiao': '🇨🇳 Xiaoxiao 🚺 (D)',
            'zf_xiaoyi': '🇨🇳 Xiaoyi 🚺 (D)',
            'zm_yunjian': '🇨🇳 Yunjian 🚹 (D)',
            'zm_yunxi': '🇨🇳 Yunxi 🚹 (D)',
            'zm_yunxia': '🇨🇳 Yunxia 🚹 (D)',
            'zm_yunyang': '🇨🇳 Yunyang 🚹 (D)',

            # Spanish
            'ef_dora': '🇪🇸 Dora 🚺',
            'em_alex': '🇪🇸 Alex 🚹',
            'em_santa': '🇪🇸 Santa 🚹',

            # French
            'ff_siwis': '🇫🇷 Siwis 🚺 (B-)',

            # Hindi
            'hf_alpha': '🇮🇳 Alpha 🚺 (C)',
            'hf_beta': '🇮🇳 Beta 🚺 (C)',
            'hm_omega': '🇮🇳 Omega 🚹 (C)',
            'hm_psi': '🇮🇳 Psi 🚹 (C)',

            # Italian
            'if_sara': '🇮🇹 Sara 🚺 (C)',
            'im_nicola': '🇮🇹 Nicola 🚹 (C)',

            # Brazilian Portuguese
            'pf_dora': '🇧🇷 Dora 🚺',
            'pm_alex': '🇧🇷 Alex 🚹',
            'pm_santa': '🇧🇷 Santa 🚹',
        }

        # Format voices, fallback to raw name if not in mapping
        formatted_voices = []
        for voice in raw_voices:
            formatted = voice_formats.get(voice, f"{voice} (Unknown)")
            formatted_voices.append(formatted)

        return formatted_voices

    except Exception:
        # Fallback to basic formatted defaults if model not loaded
        return [
            '🇺🇸 Alloy 🚺 (C)',
            '🇺🇸 Adam 🚹 (F+)',
            '🇬🇧 Emma 🚺 (B-)',
            '🇬🇧 Daniel 🚹 (D)'
        ]

def generate_audio(text: str, voice: str = None, output_path: str = "app/static/output/output.wav") -> str:
    """
    Generate audio for given text and optional voice.
    Voice parameter can be either a raw voice code (e.g., 'af_bella') or a formatted display name.
    Returns the saved file path.
    """
    if voice is None:
        voice = 'af_alloy'  # Default to American Female

    try:
        pipeline = _load_kokoro_model()

        # Convert formatted voice name to raw voice code if needed
        voice_mapping = _get_voice_mapping()
        if voice in voice_mapping:
            voice = voice_mapping[voice]  # Convert formatted name to raw code

        if voice not in pipeline.voices:
            raise ValueError(f"Voice '{voice}' not available. Available: {list(pipeline.voices.keys())}")

        # Generate audio - pipeline returns a generator yielding Result objects
        results = pipeline(text, voice=voice)

        # Collect all audio segments from the generator
        audio_segments = []
        for result in results:
            audio_segments.append(result.audio.cpu())

        # Concatenate all audio segments along time axis (dimension 0)
        if audio_segments:
            combined_audio = torch.cat(audio_segments, dim=0)
            audio_np = combined_audio.numpy()
        else:
            # Fallback if no results (shouldn't happen with valid input)
            raise RuntimeError("No audio generated from Kokoro pipeline")

        # Save the combined audio
        sf.write(output_path, audio_np, 24000)  # Kokoro uses 24kHz

        return output_path

    except Exception as e:
        raise RuntimeError(f"Kokoro generation failed: {e}")
