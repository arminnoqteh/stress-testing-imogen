import torch
from transformers import AutoModelForDocumentQuestionAnswering, AutoProcessor

def load_model(config):
    print(f"Loading model: {config['name']} ({config['dtype']})")
    model = AutoModelForDocumentQuestionAnswering.from_pretrained(
        config["name"],
        torch_dtype=getattr(torch, config["dtype"])
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(config["name"])
    return model, processor

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        # pass the file stream into safe_load
        config = yaml.safe_load(f)["model"]
    model, processor = load_model(config)
    print("Model loaded successfully")
