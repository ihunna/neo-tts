"""
Device utilities for GPU/CPU monitoring and management
"""

import torch
import time
import psutil
import platform
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information."""
    info = {
        "platform": platform.system(),
        "cpu_count": psutil.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
    }

    # PyTorch device information
    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if torch.cuda.is_available():
        info["cuda_devices"] = []
        for i in range(torch.cuda.device_count()):
            device_props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append({
                "name": device_props.name,
                "total_memory_gb": device_props.total_memory / (1024**3),
                "major": device_props.major,
                "minor": device_props.minor
            })

    # MPS (Metal Performance Shaders) for Apple Silicon
    info["mps_available"] = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

    # Default device selection
    if torch.cuda.is_available():
        info["default_device"] = "cuda"
        info["default_device_name"] = torch.cuda.get_device_name()
    elif info["mps_available"]:
        info["default_device"] = "mps"
        info["default_device_name"] = "Apple Silicon GPU"
    else:
        info["default_device"] = "cpu"
        info["default_device_name"] = f"CPU ({info['cpu_count']} cores)"

    return info

def get_optimal_device():
    """Get the optimal device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def monitor_gpu_usage() -> Dict[str, Any]:
    """Monitor current GPU usage."""
    usage = {}

    if torch.cuda.is_available():
        usage["cuda"] = []
        for i in range(torch.cuda.device_count()):
            try:
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)   # GB
                memory_free = torch.cuda.get_device_properties(i).total_memory / (1024**3) - memory_reserved

                usage["cuda"].append({
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_gb": round(memory_allocated, 2),
                    "memory_reserved_gb": round(memory_reserved, 2),
                    "memory_free_gb": round(memory_free, 2),
                    "utilization": torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else None
                })
            except Exception as e:
                logger.warning(f"Error monitoring CUDA device {i}: {e}")

    if torch.backends.mps.is_available():
        # MPS doesn't have detailed memory monitoring like CUDA
        usage["mps"] = {"available": True, "note": "MPS memory monitoring limited"}

    return usage

class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        if exc_type is None:
            logger.info(".2f")
        else:
            logger.error(".2f")

    @property
    def duration(self):
        return self.end_time - self.start_time if self.end_time else None

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def log_model_device_info(model, model_name: str):
    """Log device information for a loaded model."""
    try:
        if hasattr(model, 'parameters'):
            device = next(model.parameters()).device
            logger.info(f"{model_name} model loaded on device: {device}")
            return str(device)
        elif hasattr(model, 'device'):
            logger.info(f"{model_name} model loaded on device: {model.device}")
            return str(model.device)
        else:
            logger.info(f"{model_name} model device info not available")
            return "unknown"
    except Exception as e:
        logger.warning(f"Could not get device info for {model_name}: {e}")
        return "unknown"

def get_tensor_device_info(tensor, name: str = "tensor"):
    """Get device information for a tensor."""
    try:
        device = tensor.device
        dtype = tensor.dtype
        shape = tensor.shape
        memory_mb = tensor.numel() * tensor.element_size() / (1024**2) if tensor.numel() > 0 else 0

        return {
            "device": str(device),
            "dtype": str(dtype),
            "shape": list(shape),
            "memory_mb": round(memory_mb, 2)
        }
    except Exception as e:
        logger.warning(f"Could not get device info for {name}: {e}")
        return {"error": str(e)}
