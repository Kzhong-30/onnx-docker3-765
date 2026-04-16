import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx_app import app


class TestAPIEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)

    def test_health_check_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "live" in data["message"].lower()

    def test_predict_endpoint_valid_image(self, client):
        img = Image.new('RGB', (224, 224), color='red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert "confidence" in data
        assert "confidence_score" in data
        assert "probabilities" in data
        assert data["prediction"] in ['ant', 'bee']
        assert isinstance(data["confidence_score"], float)
        assert "ant" in data["probabilities"]
        assert "bee" in data["probabilities"]

    def test_predict_endpoint_invalid_file(self, client):
        invalid_content = b"not an image"
        
        response = client.post(
            "/predict",
            files={"file": ("test.txt", invalid_content, "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_predict_endpoint_no_file(self, client):
        response = client.post("/predict")
        assert response.status_code == 422

    def test_predict_endpoint_latency(self, client):
        import time
        
        img = Image.new('RGB', (224, 224), color='blue')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        start_time = time.time()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
        )
        latency = time.time() - start_time
        
        assert response.status_code == 200
        assert latency < 2.0, f"API endpoint latency too high: {latency:.4f}s"
