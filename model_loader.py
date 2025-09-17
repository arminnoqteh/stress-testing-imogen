import torch
from transformers import AutoModelForDocumentQuestionAnswering, AutoProcessor


def _map_dtype(dtype_str: str):
    """Map friendly dtype names to torch dtypes.

    Accepts names like 'fp32', 'float32', 'fp16', 'bf16', 'bfloat16'. Returns a torch.dtype or None.
    """
    if not dtype_str:
        return None
    name = str(dtype_str).lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": getattr(torch, "bfloat16", None),
        "bfloat16": getattr(torch, "bfloat16", None),
        # allow the canonical torch names too
        "float": torch.float32,
        "double": torch.float64,
    }
    # direct match
    if name in mapping:
        return mapping[name]

    # support strings like 'torch.float32'
    if name.startswith("torch."):
        attr = name.split(".", 1)[1]
        return getattr(torch, attr, None)

    # last resort: try getattr on torch
    return getattr(torch, name, None)


def load_model(config):
    dtype_str = config.get("dtype")
    torch_dtype = _map_dtype(dtype_str)
    print(f"Loading model: {config['name']} ({dtype_str}) -> torch_dtype={torch_dtype}")

    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForDocumentQuestionAnswering.from_pretrained(
        config["name"],
        **kwargs,
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
