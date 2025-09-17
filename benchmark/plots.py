import matplotlib.pyplot as plt
import pandas as pd
import os

def create_plots(df, output_dir="."):
    """Create visualization plots"""
    plt.figure(figsize=(15, 10))
    
    # Throughput vs Target RPM
    plt.subplot(2, 2, 1)
    plt.plot(df["Target RPM"], df["Actual RPM"], 'bo-', linewidth=2, markersize=8)
    plt.plot(df["Target RPM"], df["Target RPM"], 'r--', label='Target')
    plt.xlabel("Target RPM")
    plt.ylabel("Actual RPM")
    plt.title("Throughput vs Target RPM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Latency vs RPM
    plt.subplot(2, 2, 2)
    plt.plot(df["Target RPM"], df["Avg Latency (s)"], 'go-', label='Avg')
    plt.plot(df["Target RPM"], df["P90 Latency (s)"], 'ro-', label='P90')
    plt.plot(df["Target RPM"], df["P99 Latency (s)"], 'mo-', label='P99')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='1s Limit')
    plt.xlabel("Target RPM")
    plt.ylabel("Latency (s)")
    plt.title("Latency vs RPM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # GPU Utilization vs RPM
    plt.subplot(2, 2, 3)
    plt.plot(df["Target RPM"], df["GPU Util (%)"], 'co-', linewidth=2, markersize=8)
    plt.xlabel("Target RPM")
    plt.ylabel("GPU Utilization (%)")
    plt.title("GPU Utilization vs RPM")
    plt.grid(True, alpha=0.3)
    
    # Success Rate vs RPM
    plt.subplot(2, 2, 4)
    plt.plot(df["Target RPM"], df["Success Rate (%)"], 'ko-', linewidth=2, markersize=8)
    plt.xlabel("Target RPM")
    plt.ylabel("Success Rate (%)")
    plt.title("Success Rate vs RPM")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "t4_benchmark_plots.png"), dpi=300, bbox_inches='tight')
    plt.close()