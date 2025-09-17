import torch
import pynvml
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def get_gpu_info():
    """Get GPU information"""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**3
        compute_capability = torch.cuda.get_device_capability()
        return {
            "name": gpu_name,
            "memory_gb": gpu_memory,
            "compute_capability": compute_capability
        }
    except:
        return None

def create_synthetic_document():
    """Create a synthetic document for testing"""
    # Create a simple document-like image
    img_array = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    # Add some text-like features
    img_array[100:150, 100:500] = 200  # Header
    img_array[200:250, 100:400] = 180  # Content
    img_array[300:350, 100:450] = 190  # More content
    return Image.fromarray(img_array)

def clear_gpu_memory():
    """Clear GPU memory cache"""
    torch.cuda.empty_cache()
    import gc
    gc.collect()

def format_time(seconds):
    """Format time in human readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"