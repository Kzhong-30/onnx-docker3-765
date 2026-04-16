import pytest
import os
import time
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("MODELS_DIR", PROJECT_ROOT)
VALIDATION_DATA_DIR = os.environ.get("VALIDATION_DATA_DIR", os.path.join(PROJECT_ROOT, "hymenoptera_data", "train"))
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "my_model_quantized.onnx")
ORIGINAL_MODEL_PATH = os.path.join(MODELS_DIR, "my_model.onnx")


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    left = (256 - 224) // 2
    top = (256 - 224) // 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))
    img_array = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)


@pytest.fixture(scope="session")
def onnx_session():
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    yield session
    del session


@pytest.fixture(scope="session")
def original_onnx_session():
    if os.path.exists(ORIGINAL_MODEL_PATH):
        session = ort.InferenceSession(ORIGINAL_MODEL_PATH, providers=["CPUExecutionProvider"])
        yield session
        del session
    else:
        yield None


@pytest.fixture(scope="session")
def test_images():
    images = []
    classes = ['ants', 'bees']
    
    for class_name in classes:
        class_dir = os.path.join(VALIDATION_DATA_DIR, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for f in files[:10]:
                images.append({
                    'path': os.path.join(class_dir, f),
                    'label': class_name[:-1],
                    'class_name': class_name
                })
    
    return images


@pytest.fixture(scope="session")
def sample_image_tensor():
    for class_name in ['ants', 'bees']:
        sample_image_path = os.path.join(VALIDATION_DATA_DIR, class_name)
        if os.path.exists(sample_image_path):
            files = [f for f in os.listdir(sample_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                return preprocess_image(os.path.join(sample_image_path, files[0]))
    
    dummy_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
    return dummy_tensor


@pytest.fixture(scope="session")
def classes():
    return ['ant', 'bee']
