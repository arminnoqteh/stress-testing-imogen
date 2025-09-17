import pynvml
import time
import csv
import os
from datetime import datetime

class GPUMonitor:
    def __init__(self, interval_sec=5, output_dir="output"):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            # NVML not available; set handle to None to disable monitoring
            self.handle = None
        self.interval_sec = interval_sec
        self.output_path = os.path.join(output_dir, "gpu_metrics.csv")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gpu_util", "gpu_mem_mb", "gpu_temp"])

    def start(self, duration_sec):
        end_time = time.time() + duration_sec
        while time.time() < end_time:
            if self.handle is None:
                # write placeholder values if NVML not available
                with open(self.output_path, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now().isoformat(), 0, 0.0, 0])
                time.sleep(self.interval_sec)
                continue

            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

            # ensure numeric types for math operations
            try:
                used_bytes = int(mem.used)
            except Exception:
                # fall back to 0 if unexpected type
                used_bytes = 0

            with open(self.output_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    int(getattr(util, 'gpu', 0)),
                    used_bytes / 1024**2,
                    int(temp)
                ])
            time.sleep(self.interval_sec)

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

if __name__ == "__main__":
    monitor = GPUMonitor()
    monitor.start(60)  # Test for 60 seconds
