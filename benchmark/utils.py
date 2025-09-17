import torch
import pynvml
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from PIL import ImageDraw, ImageFont
import json
import os
import random

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


def create_id_card(name, social_no, father_name, city, province, size=(600,400)):
    """Create a synthetic ID-card-like image with the provided fields and return PIL.Image."""
    img = Image.new('RGB', size, color=(255,255,240))
    draw = ImageDraw.Draw(img)

    # Basic layout boxes
    draw.rectangle([10, 10, size[0]-10, size[1]-10], outline=(0,0,0), width=2)
    draw.rectangle([20, 20, 160, 160], outline=(0,0,0), width=1)  # photo box

    # Text positions
    x0 = 180
    y = 30
    line_h = 32

    try:
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font_b = None
        font = None

    draw.text((x0, y), "ID CARD", fill=(0,0,0), font=font_b)
    y += line_h
    draw.text((x0, y), f"Name: {name}", fill=(0,0,0), font=font); y += line_h
    draw.text((x0, y), f"Social No: {social_no}", fill=(0,0,0), font=font); y += line_h
    draw.text((x0, y), f"Father: {father_name}", fill=(0,0,0), font=font); y += line_h
    draw.text((x0, y), f"City: {city}", fill=(0,0,0), font=font); y += line_h
    draw.text((x0, y), f"Province: {province}", fill=(0,0,0), font=font); y += line_h

    # Add some noise lines
    for i in range(5):
        x1 = random.randint(20, size[0]-20)
        y1 = random.randint(20, size[1]-20)
        x2 = random.randint(20, size[0]-20)
        y2 = random.randint(20, size[1]-20)
        draw.line([x1,y1,x2,y2], fill=(200,200,200), width=1)

    return img


def save_id_card_pair(out_dir, idx, name, social_no, father_name, city, province):
    os.makedirs(out_dir, exist_ok=True)
    img = create_id_card(name, social_no, father_name, city, province)
    img_path = os.path.join(out_dir, f"id_{idx:04d}.png")
    json_path = os.path.join(out_dir, f"id_{idx:04d}.json")
    img.save(img_path)
    data = {
        "name": name,
        "social_no": social_no,
        "father_name": father_name,
        "city": city,
        "province": province,
        "image_path": img_path
    }
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    return img_path, json_path

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