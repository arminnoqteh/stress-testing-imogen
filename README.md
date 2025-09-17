# Document Processing GPU Benchmark

## Purpose
Stress-test a document verification/extraction pipeline on a **single NVIDIA T4 GPU** to determine:
- Maximum throughput (docs/minute).
- Scaling requirements for target workloads.

---

## Setup
1. **Install dependencies**:
   ```bash
   pip install torch transformers pillow pynvml pyyaml pandas matplotlib
