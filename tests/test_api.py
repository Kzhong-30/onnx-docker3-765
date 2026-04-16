import pytest
import os
import io
from PIL import Image
import numpy as np


class TestAPIEndpoints:
    
    @pytest.fixture
    def test_client(self):
        try:
            from fastapi.testclient import TestClient
            from onnx_app import app
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI test client not available")
    
    @pytest.fixture
    def sample_image_bytes(self):
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_root_endpoint(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "ONNX" in data["message"] or "live" in data["message"].lower()
    
    def test_predict_endpoint_with_valid_image(self, test_client, sample_image_bytes):
        response = test_client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ["ant", "bee"]
    
    def test_predict_endpoint_response_format(self, test_client, sample_image_bytes):
        response = test_client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert "confidence" in data
        assert "confidence_score" in data
        assert "probabilities" in data
        
        assert isinstance(data["prediction"], str)
        assert isinstance(data["confidence"], str)
        assert isinstance(data["confidence_score"], float)
        assert isinstance(data["probabilities"], dict)
    
    def test_predict_endpoint_confidence_range(self, test_client, sample_image_bytes):
        response = test_client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        
        confidence = data["confidence_score"]
        assert 0 <= confidence <= 1, f"Confidence {confidence} out of range [0, 1]"
    
    def test_predict_endpoint_probabilities_sum(self, test_client, sample_image_bytes):
        response = test_client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        
        probs = data["probabilities"]
        total_prob = probs["ant"] + probs["bee"]
        assert abs(total_prob - 1.0) < 0.01, f"Probabilities sum to {total_prob}, expected ~1.0"
    
    def test_predict_endpoint_with_invalid_file(self, test_client):
        invalid_file = io.BytesIO(b"not an image")
        response = test_client.post(
            "/predict",
            files={"file": ("test.txt", invalid_file, "text/plain")}
        )
        assert response.status_code == 400
    
    def test_predict_endpoint_with_png_image(self, test_client):
        img = Image.new('RGB', (300, 300), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = test_client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200
    
    def test_predict_endpoint_with_different_sizes(self, test_client):
        sizes = [(100, 100), (500, 500), (224, 224), (1000, 500)]
        
        for size in sizes:
            img = Image.new('RGB', size, color='green')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            response = test_client.post(
                "/predict",
                files={"file": (f"test_{size[0]}x{size[1]}.jpg", img_bytes, "image/jpeg")}
            )
            assert response.status_code == 200, f"Failed for image size {size}"
    
    def test_predict_endpoint_multiple_requests(self, test_client, sample_image_bytes):
        for i in range(10):
            sample_image_bytes.seek(0)
            response = test_client.post(
                "/predict",
                files={"file": (f"test_{i}.jpg", sample_image_bytes, "image/jpeg")}
            )
            assert response.status_code == 200
    
    def test_predict_endpoint_with_grayscale_image(self, test_client):
        img = Image.new('L', (224, 224), color=128)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = test_client.post(
            "/predict",
            files={"file": ("test_gray.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
