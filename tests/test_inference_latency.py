import pytest
import onnxruntime as ort
import numpy as np
import time
import statistics


class TestInferenceLatency:
    """推理延迟测试套件"""

    @pytest.fixture(scope="class")
    def session(self, model_path):
        """创建ONNX会话"""
        return ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

    @pytest.fixture(scope="class")
    def sample_input(self):
        """创建样本输入数据"""
        return np.random.randn(1, 3, 224, 224).astype(np.float32)

    def test_single_inference_latency(self, session, sample_input):
        """测试单次推理延迟"""
        input_name = session.get_inputs()[0].name
        
        # 预热
        for _ in range(5):
            session.run(None, {input_name: sample_input})
        
        # 测试单次推理
        start_time = time.perf_counter()
        session.run(None, {input_name: sample_input})
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        print(f"单次推理延迟: {latency_ms:.2f} ms")
        
        # 断言延迟小于1秒
        assert latency_ms < 1000, f"单次推理延迟过高: {latency_ms:.2f} ms"

    def test_multiple_inference_average_latency(self, session, sample_input):
        """测试多次推理平均延迟"""
        input_name = session.get_inputs()[0].name
        num_runs = 50
        
        # 预热
        for _ in range(10):
            session.run(None, {input_name: sample_input})
        
        # 收集延迟数据
        latencies = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            session.run(None, {input_name: sample_input})
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        avg_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"平均延迟: {avg_latency:.2f} ms")
        print(f"标准差: {std_latency:.2f} ms")
        print(f"最小延迟: {min_latency:.2f} ms")
        print(f"最大延迟: {max_latency:.2f} ms")
        
        # 断言平均延迟小于500ms
        assert avg_latency < 500, f"平均推理延迟过高: {avg_latency:.2f} ms"

    def test_batch_inference_latency(self, session):
        """测试不同批次大小的推理延迟"""
        input_name = session.get_inputs()[0].name
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            input_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            
            # 预热
            for _ in range(5):
                session.run(None, {input_name: input_data})
            
            # 测试
            latencies = []
            for _ in range(20):
                start_time = time.perf_counter()
                session.run(None, {input_name: input_data})
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)
            
            avg_latency = statistics.mean(latencies)
            latency_per_image = avg_latency / batch_size
            
            print(f"批次大小 {batch_size}: 平均延迟 {avg_latency:.2f} ms, 每张图片 {latency_per_image:.2f} ms")
            
            # 断言批次延迟合理
            assert avg_latency < batch_size * 500, f"批次大小{batch_size}的延迟过高"

    def test_throughput(self, session, sample_input):
        """测试模型吞吐量 (images/second)"""
        input_name = session.get_inputs()[0].name
        num_runs = 100
        
        # 预热
        for _ in range(10):
            session.run(None, {input_name: sample_input})
        
        # 测试吞吐量
        start_time = time.perf_counter()
        for _ in range(num_runs):
            session.run(None, {input_name: sample_input})
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        throughput = num_runs / total_time
        
        print(f"总时间: {total_time:.2f} s")
        print(f"吞吐量: {throughput:.2f} images/second")
        
        # 断言吞吐量大于1 image/second
        assert throughput > 1, f"吞吐量过低: {throughput:.2f} images/second"

    def test_p99_latency(self, session, sample_input):
        """测试P99延迟"""
        input_name = session.get_inputs()[0].name
        num_runs = 100
        
        # 预热
        for _ in range(10):
            session.run(None, {input_name: sample_input})
        
        # 收集延迟数据
        latencies = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            session.run(None, {input_name: sample_input})
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        latencies.sort()
        p99_index = int(len(latencies) * 0.99)
        p99_latency = latencies[p99_index]
        
        print(f"P99延迟: {p99_latency:.2f} ms")
        
        # 断言P99延迟小于1秒
        assert p99_latency < 1000, f"P99延迟过高: {p99_latency:.2f} ms"

    @pytest.mark.parametrize("input_size", [
        (1, 3, 224, 224),
        (1, 3, 256, 256),
        (1, 3, 192, 192),
    ])
    def test_different_input_sizes_latency(self, session, input_size):
        """测试不同输入尺寸的延迟"""
        input_name = session.get_inputs()[0].name
        input_data = np.random.randn(*input_size).astype(np.float32)
        
        # 如果输入尺寸不匹配模型期望的尺寸，需要调整
        expected_shape = session.get_inputs()[0].shape
        if input_size != tuple(expected_shape if expected_shape[0] is None else expected_shape):
            pytest.skip(f"输入尺寸{input_size}与模型期望的尺寸不匹配")
        
        # 预热
        for _ in range(5):
            session.run(None, {input_name: input_data})
        
        # 测试
        latencies = []
        for _ in range(20):
            start_time = time.perf_counter()
            session.run(None, {input_name: input_data})
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        avg_latency = statistics.mean(latencies)
        print(f"输入尺寸 {input_size}: 平均延迟 {avg_latency:.2f} ms")
        
        assert avg_latency < 1000, f"输入尺寸{input_size}的延迟过高"
