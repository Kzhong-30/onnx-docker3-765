import pytest
import os
import onnxruntime as ort


MODEL_PATH = os.environ.get("MODELS_DIR", "/code") + "/my_model_quantized.onnx"
ORIGINAL_MODEL_PATH = os.environ.get("MODELS_DIR", "/code") + "/my_model.onnx"


class TestModelLoading:
    
    def test_model_file_exists(self):
        assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    
    def test_model_file_size(self):
        file_size = os.path.getsize(MODEL_PATH)
        assert file_size > 0, "Model file is empty"
        assert file_size < 100 * 1024 * 1024, f"Model file too large: {file_size} bytes"
    
    def test_model_load_success(self):
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        assert session is not None, "Failed to load ONNX model"
        del session
    
    def test_model_input_shape(self, onnx_session):
        input_info = onnx_session.get_inputs()[0]
        input_shape = input_info.shape
        assert len(input_shape) == 4, f"Expected 4D input, got {len(input_shape)}D"
        assert input_shape[0] in [None, 1, 'batch_size'], f"Unexpected batch dimension: {input_shape[0]}"
        assert input_shape[1] == 3, f"Expected 3 channels, got {input_shape[1]}"
        assert input_shape[2] == 224, f"Expected height 224, got {input_shape[2]}"
        assert input_shape[3] == 224, f"Expected width 224, got {input_shape[3]}"
    
    def test_model_output_shape(self, onnx_session):
        output_info = onnx_session.get_outputs()[0]
        output_shape = output_info.shape
        assert len(output_shape) == 2, f"Expected 2D output, got {len(output_shape)}D"
        assert output_shape[1] == 2, f"Expected 2 classes, got {output_shape[1]}"
    
    def test_model_input_name(self, onnx_session):
        input_name = onnx_session.get_inputs()[0].name
        assert input_name == 'input', f"Unexpected input name: {input_name}"
    
    def test_model_output_name(self, onnx_session):
        output_name = onnx_session.get_outputs()[0].name
        assert output_name is not None, "Output name is None"
    
    def test_model_providers(self, onnx_session):
        available_providers = ort.get_available_providers()
        assert "CPUExecutionProvider" in available_providers, "CPUExecutionProvider not available"
    
    def test_model_metadata(self, onnx_session):
        metadata = onnx_session.get_modelmeta()
        assert metadata is not None, "Model metadata is None"
    
    def test_original_model_exists(self):
        if os.path.exists(ORIGINAL_MODEL_PATH):
            assert os.path.getsize(ORIGINAL_MODEL_PATH) > 0, "Original model file is empty"
    
    def test_quantized_model_smaller(self):
        if os.path.exists(ORIGINAL_MODEL_PATH) and os.path.exists(MODEL_PATH):
            original_size = os.path.getsize(ORIGINAL_MODEL_PATH)
            quantized_size = os.path.getsize(MODEL_PATH)
            assert quantized_size <= original_size, \
                f"Quantized model ({quantized_size} bytes) should be smaller than original ({original_size} bytes)"
