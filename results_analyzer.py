import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml

def analyze_results(config):
    results_path = os.path.join(config["benchmark"]["output_dir"], "benchmark_results.csv")
    df = pd.read_csv(results_path)

    # Print report
    print("\n--- Benchmark Results ---")
    print(df.to_markdown(index=False))

    # Plot throughput vs latency
    plt.figure(figsize=(10, 5))
    plt.plot(df["batch_size"], df["docs_per_minute"], marker='o', label="Throughput")
    plt.axhline(y=config["benchmark"]["max_latency"] * 60, color='r', linestyle='--', label="Max Latency")
    plt.xlabel("Batch Size")
    plt.ylabel("Docs per Minute")
    plt.title("Throughput vs Batch Size")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config["benchmark"]["output_dir"], "throughput_vs_batch.png"))
    plt.show()

    # Generate scaling guide
    max_throughput = df["docs_per_minute"].max()
    print(f"\n--- Scaling Guide ---")
    print(f"Max throughput on T4: {max_throughput:.1f} docs/minute")
    print("Required GPUs for target throughput:")
    for target in [500, 1000, 2000, 5000]:
        gpus = target / max_throughput
        print(f"- {target} docs/min: {gpus:.1f} x T4")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        # load full config dict from file
        config = yaml.safe_load(f)
    analyze_results(config)
