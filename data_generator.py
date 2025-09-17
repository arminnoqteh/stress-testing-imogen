import os
import random
from PIL import Image, ImageDraw, ImageFont
import yaml
import numpy as np

def generate_fake_id_card(save_path, fields=["Name", "ID", "Date"]):
    img = Image.new("RGB", (800, 500), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Add random text fields
    draw.text((50, 50), f"Name: {random.choice(['Alice', 'Bob', 'Charlie', 'David'])}", fill=(0, 0, 0), font=font)
    draw.text((50, 100), f"ID: {random.randint(100000, 999999)}", fill=(0, 0, 0), font=font)
    draw.text((50, 150), f"Date: {random.choice(['2023-01-01', '2024-05-15', '2025-12-31'])}", fill=(0, 0, 0), font=font)

    # Add a fake signature
    for _ in range(10):
        x1, y1 = random.randint(400, 700), random.randint(300, 400)
        x2, y2 = x1 + random.randint(-20, 20), y1 + random.randint(-20, 20)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=2)

    img.save(save_path)

def generate_documents(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    for i in range(config["num_docs"]):
        generate_fake_id_card(
            os.path.join(config["output_dir"], f"fake_id_{i}.png"),
            fields=config["fields"]
        )

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        # pass file stream into safe_load
        config = yaml.safe_load(f)["data_generator"]
    generate_documents(config)
    print(f"Generated {config['num_docs']} synthetic documents to {config['output_dir']}")
