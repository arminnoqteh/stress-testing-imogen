import os
import time
import yaml
import random
import torch
from PIL import Image
from data_generator import generate_documents
from model_loader import load_model
from gpu_monitor import GPUMonitor
import pandas as pd

def run_benchmark(config):
    # Generate documents
    generate_documents(config["data_generator"])

    # Load model
    model, processor = load_model(config["model"])

    # Prepare sample paths
    sample_dir = config["data_generator"]["output_dir"]
    sample_paths = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(".png")]

    results = []
    for batch_size in config["benchmark"]["batch_sizes"]:
        print(f"\nTesting batch size: {batch_size}")

        # Start GPU monitor (will write metrics to CSV in the background)
        monitor = GPUMonitor(
            interval_sec=config["gpu_monitor"]["interval_sec"],
            output_dir=config["benchmark"]["output_dir"]
        )

        # Run benchmark
        start_time = time.time()
        processed = 0
        latencies = []

        # start() is blocking; run it alongside the benchmark if needed or run monitoring in the same loop
        # For simplicity, we'll run monitoring in the background by spawning a thread if desired. If pynvml is not set up,
        # start() will still run in this thread. To avoid adding threading as a dependency here, we'll call start with duration 1
        # in a non-blocking manner by using a simple approach: start monitoring in a separate thread only when available.
        try:
            import threading
            monitor_thread = threading.Thread(target=monitor.start, args=(config["benchmark"]["duration_min"] * 60,), daemon=True)
            monitor_thread.start()
        except Exception:
            # fallback: run in same thread (blocking) which will still collect metrics
            monitor.start(config["benchmark"]["duration_min"] * 60)

        while time.time() - start_time < config["benchmark"]["duration_min"] * 60:
            batch_start = time.time()
            batch_paths = random.sample(sample_paths, min(batch_size, len(sample_paths)))

            for path in batch_paths:
                image = Image.open(path)
                inputs = processor(image, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    _ = model(**inputs)

            batch_time = time.time() - batch_start
            latencies.append(batch_time)
            processed += batch_size

        # Calculate metrics
        duration_min = (time.time() - start_time) / 60
        rpm = (processed / duration_min) if duration_min > 0 else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        results.append({
            "batch_size": batch_size,
            "docs_per_minute": rpm,
            "avg_latency": avg_latency
        })

        print(f"Results: {rpm:.1f} docs/min | Latency: {avg_latency:.3f}s")

        if avg_latency > config["benchmark"]["max_latency"]:
            print(f"⚠️ Max latency exceeded at batch size {batch_size}")
            break

    # Save results
    df = pd.DataFrame(results)
    os.makedirs(config["benchmark"]["output_dir"], exist_ok=True)
    df.to_csv(os.path.join(config["benchmark"]["output_dir"], "benchmark_results.csv"), index=False)
    return df

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    results = run_benchmark(config)
    print("\nBenchmark completed. Results saved to output/benchmark_results.csv")
