import torch
from transformers import AutoModelForDocumentQuestionAnswering, AutoProcessor
import pynvml
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import get_gpu_info, create_synthetic_document, clear_gpu_memory, format_time
from .plots import create_plots

class T4Benchmark:
    def __init__(self, model_name="microsoft/layoutlmv3-base"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.gpu_info = get_gpu_info()
        
    def load_model(self):
        """Load the model"""
        print(f"\nLoading model: {self.model_name}")
        self.model = AutoModelForDocumentQuestionAnswering.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to("cuda")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print("Model loaded successfully!")
        
    def infer_single_document(self, document, doc_id):
        """Process a single document and return latency"""
        try:
            start_time = time.time()
            # Processor may return a dict/BatchEncoding; move tensors inside to CUDA
            inputs = self.processor(document, return_tensors="pt")

            # Recursive mover to handle nested lists/tuples/dicts of tensors
            def move_to_device(obj, device="cuda"):
                if torch.is_tensor(obj):
                    return obj.to(device)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v, device) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    converted = [move_to_device(v, device) for v in obj]
                    return type(obj)(converted)
                else:
                    return obj

            inputs = move_to_device(inputs, device="cuda")

            # Debug: briefly show what keys/types are being passed to the model
            try:
                keys_info = {k: type(v).__name__ for k, v in inputs.items()} if isinstance(inputs, dict) else type(inputs).__name__
                print(f"  Doc {doc_id} inputs: {keys_info}")
            except Exception:
                pass

            with torch.no_grad():
                outputs = self.model(**inputs)
            latency = time.time() - start_time
            return {"doc_id": doc_id, "latency": latency, "success": True}
        except Exception as e:
            # Print full traceback for visibility in logs
            import traceback
            tb = traceback.format_exc()
            print(f"  Inference error for doc {doc_id}: {e}\n{tb}")
            return {"doc_id": doc_id, "latency": 0, "success": False, "error": str(e), "traceback": tb}
    
    def stress_test(self, target_rpm, duration_min=2.0):
        """Run stress test. `duration_min` can be a float to allow short debug runs."""
        target_requests = int(target_rpm * duration_min)
        actual_requests = 0
        latencies = []
        errors = []
        
        # Calculate inter-request interval to achieve target RPM
        if target_rpm > 0:
            inter_request_interval = 60.0 / target_rpm
        else:
            inter_request_interval = 0.1
        
        print(f"  Target: {target_rpm} RPM, {target_requests} total requests")
        print(f"  Inter-request interval: {inter_request_interval:.3f}s")
        
        try:
            # Ensure model is loaded before starting the stress test
            if self.model is None or self.processor is None:
                try:
                    self.load_model()
                except Exception as e:
                    print(f"  Failed to load model before stress test: {e}")
                    return {
                        "target_rpm": target_rpm,
                        "actual_rpm": 0,
                        "total_requests": 0,
                        "avg_latency": 0,
                        "p50_latency": 0,
                        "p90_latency": 0,
                        "p99_latency": 0,
                        "max_latency": 0,
                        "gpu_util": 0,
                        "memory_used": 0,
                        "error_count": 1,
                        "success_rate": 0,
                        "error": str(e)
                    }

            # Create document pool
            document_pool = [create_synthetic_document() for _ in range(10)]

            # Use ThreadPoolExecutor for concurrent processing
            # Ensure at least one worker
            max_workers = max(1, min(8, target_rpm // 10))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                start_time = time.time()
                
                # Submit tasks in batches
                futures = []
                
                for i in range(target_requests):
                    request_time = start_time + i * inter_request_interval
                    delay = max(0, request_time - time.time())
                    
                    doc = document_pool[i % len(document_pool)]
                    # Submit the inference job directly (avoid lambda capture issues)
                    future = executor.submit(self.infer_single_document, doc, i)
                    futures.append(future)
                    
                    if delay > 0:
                        time.sleep(min(delay, 0.1))
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        # Unexpected exception from the future
                        errors.append({"doc_id": None, "latency": 0, "success": False, "error": str(e)})
                        result = None

                    if result:
                        if result["success"]:
                            latencies.append(result["latency"])
                            actual_requests += 1
                        else:
                            errors.append(result)

                    # Stop collecting once test duration has elapsed
                    if time.time() - start_time > duration_min * 60:
                        break
            
            # Calculate statistics
            # Debug: print submission / result summary
            try:
                print(f"  Submitted futures: {len(futures)} | Successes: {actual_requests} | Errors: {len(errors)}")
                if errors:
                    print("  Sample errors (up to 5):")
                    for err in errors[:5]:
                        # err may be an exception dict or object
                        if isinstance(err, dict):
                            print(f"    - doc_id={err.get('doc_id')} error={err.get('error')}")
                        else:
                            print(f"    - {err}")
            except Exception:
                pass
            if latencies:
                avg_latency = np.mean(latencies)
                p50_latency = np.percentile(latencies, 50)
                p90_latency = np.percentile(latencies, 90)
                p99_latency = np.percentile(latencies, 99)
                max_latency = np.max(latencies)
            else:
                avg_latency = p50_latency = p90_latency = p99_latency = max_latency = 0
            
            actual_rpm = actual_requests / duration_min
            
            # Get final GPU metrics
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_util = util.gpu
                memory_used = mem.used / 1024**2
            except:
                gpu_util = 0
                memory_used = 0
            
            return {
                "target_rpm": target_rpm,
                "actual_rpm": actual_rpm,
                "total_requests": actual_requests,
                "avg_latency": avg_latency,
                "p50_latency": p50_latency,
                "p90_latency": p90_latency,
                "p99_latency": p99_latency,
                "max_latency": max_latency,
                "gpu_util": gpu_util,
                "memory_used": memory_used,
                "error_count": len(errors),
                "success_rate": actual_requests / target_requests if target_requests > 0 else 0
            }
        
        except Exception as e:
            print(f"  Error during stress test: {str(e)}")
            return {
                "target_rpm": target_rpm,
                "actual_rpm": 0,
                "total_requests": 0,
                "avg_latency": 0,
                "p50_latency": 0,
                "p90_latency": 0,
                "p99_latency": 0,
                "max_latency": 0,
                "gpu_util": 0,
                "memory_used": 0,
                "error_count": 1,
                "success_rate": 0,
                "error": str(e)
            }
        
        finally:
            clear_gpu_memory()
    
    def run_benchmark(self):
        """Run the complete benchmark"""
        print("\n" + "="*50)
        print("T4 GPU DOCUMENT PIPELINE BENCHMARK")
        print("="*50)
        
        if self.gpu_info:
            print(f"\nBenchmark System:")
            print(f"  GPU: {self.gpu_info['name']}")
            print(f"  GPU Memory: {self.gpu_info['memory_gb']:.1f} GB")
            print(f"  Compute Capability: {self.gpu_info['compute_capability']}")
        
        print(f"\nModel: {self.model_name} (FP32)")
        
        # Test points
        test_points = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        results = []
        
        for rpm in test_points:
            print(f"\n{'='*20} Testing {rpm} RPM {'='*20}")
            
            # Run stress test
            result = self.stress_test(rpm, duration_min=2)
            
            # Display results
            print(f"\nResults for {rpm} RPM:")
            print(f"  Target: {rpm} RPM | Actual: {result['actual_rpm']:.1f} RPM")
            print(f"  Avg Latency: {format_time(result['avg_latency'])} | P90: {format_time(result['p90_latency'])} | P99: {format_time(result['p99_latency'])}")
            print(f"  GPU Util: {result['gpu_util']}% | Memory: {result['memory_used']:.1f}MB")
            print(f"  Success Rate: {result['success_rate']*100:.1f}%")
            
            results.append(result)
            
            # Check failure conditions
            if result['p90_latency'] > 1.0:
                print(f"\n⚠️ P90 latency exceeded 1s at {rpm} RPM!")
                break
            
            if self.gpu_info and result['memory_used'] > self.gpu_info['memory_gb'] * 1024 * 0.95:
                print(f"\n⚠️ GPU memory exhausted at {rpm} RPM!")
                break
            
            # Small delay between tests
            time.sleep(5)
        
        return results
    
    def generate_report(self, results):
        """Generate benchmark report"""
        if not results:
            print("No results to report!")
            return None
        
        # Create DataFrame
        df_data = []
        for result in results:
            df_data.append({
                "Target RPM": result['target_rpm'],
                "Actual RPM": result['actual_rpm'],
                "Success Rate (%)": result['success_rate'] * 100,
                "Avg Latency (s)": result['avg_latency'],
                "P50 Latency (s)": result['p50_latency'],
                "P90 Latency (s)": result['p90_latency'],
                "P99 Latency (s)": result['p99_latency'],
                "Max Latency (s)": result['max_latency'],
                "GPU Util (%)": result['gpu_util'],
                "GPU Mem (MB)": result['memory_used'],
                "Error Count": result['error_count']
            })
        
        df = pd.DataFrame(df_data)
        
        # Print report
        print("\n" + "="*60)
        print("T4 GPU DOCUMENT PIPELINE BENCHMARK REPORT")
        print("="*60)
        
        if self.gpu_info:
            print(f"\nBenchmark System:")
            print(f"  GPU: {self.gpu_info['name']}")
            print(f"  GPU Memory: {self.gpu_info['memory_gb']:.1f} GB")
            print(f"  Compute Capability: {self.gpu_info['compute_capability']}")
        
        print(f"\nModel: {self.model_name} (FP32)")
        print(f"Test Duration: 2 minutes per load level")
        
        print("\nDetailed Results:")
        print(df.to_markdown(index=False, floatfmt=".2f"))
        
        # Save to CSV
        df.to_csv("t4_benchmark_results.csv", index=False)
        print(f"\nResults saved to: t4_benchmark_results.csv")
        
        # Create plots
        create_plots(df)
        print(f"Plots saved to: t4_benchmark_plots.png")
        
        # Find saturation point
        saturation_point = None
        for i, result in enumerate(results):
            if result['p90_latency'] > 1.0:
                saturation_point = result['target_rpm']
                break
        
        if saturation_point:
            print(f"\n🚨 FAILURE POINT: T4 saturates at {saturation_point} RPM with P99 latency >1s")
            print(f"📊 Max sustainable throughput: {results[i-1]['actual_rpm']:.0f} RPM")
        else:
            print(f"\n✅ All tests passed within latency limits")
            print(f"📊 Max tested throughput: {results[-1]['actual_rpm']:.0f} RPM")
        
        # Generate scaling recommendations
        print(f"\n📈 SCALING RECOMMENDATIONS:")
        if saturation_point:
            max_rpm = results[i-1]['actual_rpm']
            print(f"  For {max_rpm} RPM: 1x T4 GPU")
            print(f"  For {max_rpm * 2} RPM: 2x T4 GPUs")
            print(f"  For {max_rpm * 4} RPM: 4x T4 GPUs")
            print(f"  For {max_rpm * 8} RPM: 8x T4 GPUs")
        else:
            max_rpm = results[-1]['actual_rpm']
            print(f"  For {max_rpm} RPM: 1x T4 GPU")
            print(f"  For {max_rpm * 2} RPM: 2x T4 GPUs")
            print(f"  For {max_rpm * 4} RPM: 4x T4 GPUs")
            print(f"  For {max_rpm * 8} RPM: 8x T4 GPUs")
        
        return df

if __name__ == "__main__":
    # Create benchmark instance
    benchmark = T4Benchmark()
    
    # Load model
    import sys

    # If user passes --debug, run a quick smoke test (5 requests) and exit
    if "--debug" in sys.argv:
        print("Running debug smoke test (5 requests)...")
        # ensure model loaded
        benchmark.load_model()
        # Run 5 synchronous inferences to quickly surface any errors
        docs = [create_synthetic_document() for _ in range(5)]
        dbg_results = []
        for i, d in enumerate(docs):
            res = benchmark.infer_single_document(d, i)
            print(f"  Sync inference result: {res}")
            dbg_results.append(res)
        print("Debug results summary:")
        print({
            "submitted": len(dbg_results),
            "successes": sum(1 for r in dbg_results if r.get("success")),
            "errors": [r.get("error") for r in dbg_results if not r.get("success")]
        })
        sys.exit(0)

    benchmark.load_model()

    # Run benchmark
    results = benchmark.run_benchmark()
    
    # Generate report
    if results:
        df = benchmark.generate_report(results)
        
        # Print final summary
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)
        if df is not None:
            print(f"🎯 Max sustainable throughput: {df['Actual RPM'].max():.0f} RPM")
            print(f"📊 Max GPU utilization: {df['GPU Util (%)'].max():.0f}%")
            print(f"⚡ Min average latency: {format_time(df['Avg Latency (s)'].min())}")
            print(f"📈 Scaling factor: 1x T4 = {df['Actual RPM'].max():.0f} RPM")
        else:
            print("No dataframe available to summarise results.")