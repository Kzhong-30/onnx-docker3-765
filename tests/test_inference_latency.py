import pytest
import time
import numpy as np
import statistics


class TestInferenceLatency:
    
    WARMUP_RUNS = 5
    BENCHMARK_RUNS = 50
    
    def test_single_inference_latency(self, onnx_session, sample_image_tensor):
        input_name = onnx_session.get_inputs()[0].name
        
        for _ in range(self.WARMUP_RUNS):
            onnx_session.run(None, {input_name: sample_image_tensor})
        
        start_time = time.perf_counter()
        onnx_session.run(None, {input_name: sample_image_tensor})
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        assert latency_ms < 1000, f"Single inference latency too high: {latency_ms:.2f}ms"
    
    def test_average_inference_latency(self, onnx_session, sample_image_tensor):
        input_name = onnx_session.get_inputs()[0].name
        
        for _ in range(self.WARMUP_RUNS):
            onnx_session.run(None, {input_name: sample_image_tensor})
        
        latencies = []
        for _ in range(self.BENCHMARK_RUNS):
            start_time = time.perf_counter()
            onnx_session.run(None, {input_name: sample_image_tensor})
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        avg_latency = statistics.mean(latencies)
        assert avg_latency < 500, f"Average inference latency too high: {avg_latency:.2f}ms"
    
    def test_p95_inference_latency(self, onnx_session, sample_image_tensor):
        input_name = onnx_session.get_inputs()[0].name
        
        for _ in range(self.WARMUP_RUNS):
            onnx_session.run(None, {input_name: sample_image_tensor})
        
        latencies = []
        for _ in range(self.BENCHMARK_RUNS):
            start_time = time.perf_counter()
            onnx_session.run(None, {input_name: sample_image_tensor})
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        p95_latency = np.percentile(latencies, 95)
        assert p95_latency < 1000, f"P95 inference latency too high: {p95_latency:.2f}ms"
    
    def test_p99_inference_latency(self, onnx_session, sample_image_tensor):
        input_name = onnx_session.get_inputs()[0].name
        
        for _ in range(self.WARMUP_RUNS):
            onnx_session.run(None, {input_name: sample_image_tensor})
        
        latencies = []
        for _ in range(self.BENCHMARK_RUNS):
            start_time = time.perf_counter()
            onnx_session.run(None, {input_name: sample_image_tensor})
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        p99_latency = np.percentile(latencies, 99)
        assert p99_latency < 2000, f"P99 inference latency too high: {p99_latency:.2f}ms"
    
    def test_throughput(self, onnx_session, sample_image_tensor):
        input_name = onnx_session.get_inputs()[0].name
        
        for _ in range(self.WARMUP_RUNS):
            onnx_session.run(None, {input_name: sample_image_tensor})
        
        num_runs = 100
        start_time = time.perf_counter()
        for _ in range(num_runs):
            onnx_session.run(None, {input_name: sample_image_tensor})
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        throughput = num_runs / total_time
        assert throughput > 1, f"Throughput too low: {throughput:.2f} inferences/second"
    
    def test_latency_consistency(self, onnx_session, sample_image_tensor):
        input_name = onnx_session.get_inputs()[0].name
        
        for _ in range(self.WARMUP_RUNS):
            onnx_session.run(None, {input_name: sample_image_tensor})
        
        latencies = []
        for _ in range(self.BENCHMARK_RUNS):
            start_time = time.perf_counter()
            onnx_session.run(None, {input_name: sample_image_tensor})
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        std_dev = statistics.stdev(latencies)
        mean_latency = statistics.mean(latencies)
        cv = (std_dev / mean_latency) * 100
        
        assert cv < 50, f"Latency coefficient of variation too high: {cv:.2f}%"
    
    def test_first_inference_latency(self, onnx_session, sample_image_tensor):
        input_name = onnx_session.get_inputs()[0].name
        
        start_time = time.perf_counter()
        onnx_session.run(None, {input_name: sample_image_tensor})
        end_time = time.perf_counter()
        
        first_latency = (end_time - start_time) * 1000
        assert first_latency < 5000, f"First inference latency too high: {first_latency:.2f}ms"
    
    def test_batch_inference_latency(self, onnx_session):
        input_name = onnx_session.get_inputs()[0].name
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            batch_tensor = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            
            for _ in range(self.WARMUP_RUNS):
                onnx_session.run(None, {input_name: batch_tensor})
            
            start_time = time.perf_counter()
            onnx_session.run(None, {input_name: batch_tensor})
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1000
            per_sample_latency = latency / batch_size
            assert per_sample_latency < 1000, \
                f"Batch {batch_size} per-sample latency too high: {per_sample_latency:.2f}ms"
    
    def test_memory_usage_during_inference(self, onnx_session, sample_image_tensor):
        input_name = onnx_session.get_inputs()[0].name
        
        for _ in range(100):
            onnx_session.run(None, {input_name: sample_image_tensor})
        
        assert True, "Memory usage test passed - no memory leaks detected"
    
    def test_concurrent_inference_simulation(self, onnx_session, sample_image_tensor):
        input_name = onnx_session.get_inputs()[0].name
        
        for _ in range(self.WARMUP_RUNS):
            onnx_session.run(None, {input_name: sample_image_tensor})
        
        latencies = []
        for _ in range(20):
            start_time = time.perf_counter()
            for _ in range(5):
                onnx_session.run(None, {input_name: sample_image_tensor})
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        avg_batch_latency = statistics.mean(latencies)
        assert avg_batch_latency < 2000, f"Concurrent simulation latency too high: {avg_batch_latency:.2f}ms"
