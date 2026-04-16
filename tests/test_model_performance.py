import os
import time
import pytest
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
except Exception:
    ONNXRUNTIME_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not ONNXRUNTIME_AVAILABLE,
    reason="onnxruntime not available in this environment"
)


class TestModelLoading:
    def test_model_file_exists(self):
        model_path = "my_model_quantized.onnx"
        assert os.path.exists(model_path), f"Model file not found: {model_path}"

    @pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not available")
    def test_model_loading_success(self):
        model_path = "my_model_quantized.onnx"
        try:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            assert session is not None, "Failed to create inference session"
        except Exception as e:
            pytest.fail(f"Model loading failed with error: {str(e)}")

    @pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not available")
    def test_model_input_output_shape(self):
        model_path = "my_model_quantized.onnx"
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        
        assert input_shape == [None, 3, 224, 224], f"Unexpected input shape: {input_shape}"
        assert output_shape == [None, 2], f"Unexpected output shape: {output_shape}"

    @pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not available")
    def test_model_input_name(self):
        model_path = "my_model_quantized.onnx"
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        assert input_name == "input", f"Unexpected input name: {input_name}"


class TestInferenceLatency:
    @pytest.fixture(scope="class")
    def inference_session(self):
        if not ONNXRUNTIME_AVAILABLE:
            pytest.skip("onnxruntime not available")
        model_path = "my_model_quantized.onnx"
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    @pytest.fixture(scope="class")
    def sample_image_tensor(self):
        transforms_list = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img = Image.new('RGB', (224, 224), color='red')
        return transforms_list(img).unsqueeze(0).numpy()

    def test_single_inference_latency(self, inference_session, sample_image_tensor):
        start_time = time.time()
        inference_session.run(None, {'input': sample_image_tensor})
        latency = time.time() - start_time
        
        assert latency < 1.0, f"Single inference too slow: {latency:.4f}s"
        print(f"Single inference latency: {latency:.4f}s")

    def test_average_inference_latency(self, inference_session, sample_image_tensor):
        num_runs = 50
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.time()
            inference_session.run(None, {'input': sample_image_tensor})
            latencies.append(time.time() - start_time)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"Average latency over {num_runs} runs: {avg_latency:.4f}s")
        print(f"P95 latency: {p95_latency:.4f}s")
        print(f"P99 latency: {p99_latency:.4f}s")
        
        assert avg_latency < 0.5, f"Average latency too high: {avg_latency:.4f}s"
        assert p95_latency < 0.8, f"P95 latency too high: {p95_latency:.4f}s"

    def test_throughput(self, inference_session, sample_image_tensor):
        duration = 5
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < duration:
            inference_session.run(None, {'input': sample_image_tensor})
            count += 1
        
        throughput = count / duration
        print(f"Throughput: {throughput:.2f} inferences/second ({count} in {duration}s)")
        
        assert throughput > 1.0, f"Throughput too low: {throughput:.2f} inferences/second"


class TestModelAccuracy:
    @pytest.fixture(scope="class")
    def inference_session(self):
        if not ONNXRUNTIME_AVAILABLE:
            pytest.skip("onnxruntime not available")
        model_path = "my_model_quantized.onnx"
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    @pytest.fixture(scope="class")
    def image_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_validation_images(self, class_name):
        val_path = f"hymenoptera_data/val/{class_name}"
        if os.path.exists(val_path):
            return [os.path.join(val_path, f) for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        return []

    def predict_image(self, session, transforms, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = transforms(img).unsqueeze(0).numpy()
        outputs = session.run(None, {'input': img_tensor})
        logits = outputs[0]
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        predicted_idx = np.argmax(probabilities)
        return ['ant', 'bee'][predicted_idx]

    def test_accuracy_on_ants(self, inference_session, image_transforms):
        ant_images = self.get_validation_images('ants')
        if not ant_images:
            pytest.skip("No ant validation images found")
        
        correct = 0
        for img_path in ant_images:
            prediction = self.predict_image(inference_session, image_transforms, img_path)
            if prediction == 'ant':
                correct += 1
        
        accuracy = correct / len(ant_images)
        print(f"Ant classification accuracy: {accuracy:.2%} ({correct}/{len(ant_images)})")
        
        assert accuracy >= 0.7, f"Ant accuracy too low: {accuracy:.2%}"

    def test_accuracy_on_bees(self, inference_session, image_transforms):
        bee_images = self.get_validation_images('bees')
        if not bee_images:
            pytest.skip("No bee validation images found")
        
        correct = 0
        for img_path in bee_images:
            prediction = self.predict_image(inference_session, image_transforms, img_path)
            if prediction == 'bee':
                correct += 1
        
        accuracy = correct / len(bee_images)
        print(f"Bee classification accuracy: {accuracy:.2%} ({correct}/{len(bee_images)})")
        
        assert accuracy >= 0.7, f"Bee accuracy too low: {accuracy:.2%}"

    def test_overall_accuracy(self, inference_session, image_transforms):
        all_images = []
        all_labels = []
        
        ant_images = self.get_validation_images('ants')
        bee_images = self.get_validation_images('bees')
        
        all_images.extend(ant_images)
        all_images.extend(bee_images)
        all_labels.extend(['ant'] * len(ant_images))
        all_labels.extend(['bee'] * len(bee_images))
        
        if not all_images:
            pytest.skip("No validation images found")
        
        correct = 0
        for img_path, label in zip(all_images, all_labels):
            prediction = self.predict_image(inference_session, image_transforms, img_path)
            if prediction == label:
                correct += 1
        
        accuracy = correct / len(all_images)
        print(f"Overall classification accuracy: {accuracy:.2%} ({correct}/{len(all_images)})")
        
        assert accuracy >= 0.75, f"Overall accuracy too low: {accuracy:.2%}"

    def test_prediction_confidence(self, inference_session, image_transforms):
        bee_images = self.get_validation_images('bees')[:5]
        if not bee_images:
            pytest.skip("No bee validation images found")
        
        for img_path in bee_images:
            img = Image.open(img_path).convert('RGB')
            img_tensor = image_transforms(img).unsqueeze(0).numpy()
            outputs = inference_session.run(None, {'input': img_tensor})
            logits = outputs[0]
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            confidence = float(np.max(probabilities))
            
            assert confidence > 0.5, f"Low confidence prediction: {confidence:.2%} for {img_path}"
