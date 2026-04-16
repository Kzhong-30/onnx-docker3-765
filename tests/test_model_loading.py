import pytest
import onnxruntime as ort
import os


class TestModelLoading:
    """模型加载测试套件"""

    def test_model_file_exists(self, model_path):
        """测试模型文件是否存在"""
        assert os.path.exists(model_path), f"模型文件不存在: {model_path}"

    def test_model_file_not_empty(self, model_path):
        """测试模型文件不为空"""
        file_size = os.path.getsize(model_path)
        assert file_size > 0, f"模型文件为空: {model_path}"
        print(f"模型文件大小: {file_size / 1024 / 1024:.2f} MB")

    def test_model_load_success(self, model_path):
        """测试模型能否成功加载"""
        try:
            session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            assert session is not None
        except Exception as e:
            pytest.fail(f"模型加载失败: {str(e)}")

    def test_model_input_shape(self, model_path):
        """测试模型输入形状是否正确"""
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        input_shape = session.get_inputs()[0].shape
        assert input_shape[0] in [None, 1], f"批次维度应为None或1, 实际为{input_shape[0]}"
        assert input_shape[1] == 3, f"通道数应为3, 实际为{input_shape[1]}"
        assert input_shape[2] == 224, f"高度应为224, 实际为{input_shape[2]}"
        assert input_shape[3] == 224, f"宽度应为224, 实际为{input_shape[3]}"

    def test_model_output_shape(self, model_path, classes):
        """测试模型输出形状是否正确"""
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        output_shape = session.get_outputs()[0].shape
        assert output_shape[1] == len(classes), f"输出类别数应为{len(classes)}, 实际为{output_shape[1]}"

    def test_model_providers_available(self, model_path):
        """测试可用的执行提供程序"""
        available_providers = ort.get_available_providers()
        assert "CPUExecutionProvider" in available_providers, "CPUExecutionProvider不可用"
        print(f"可用执行提供程序: {available_providers}")

    @pytest.mark.parametrize("provider", ["CPUExecutionProvider"])
    def test_model_load_with_different_providers(self, model_path, provider):
        """测试使用不同执行提供程序加载模型"""
        available_providers = ort.get_available_providers()
        if provider not in available_providers:
            pytest.skip(f"{provider}不可用")
        
        try:
            session = ort.InferenceSession(
                model_path,
                providers=[provider]
            )
            assert session is not None
        except Exception as e:
            pytest.fail(f"使用{provider}加载模型失败: {str(e)}")

    def test_model_metadata(self, model_path):
        """测试模型元数据"""
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        
        modelmeta = session.get_modelmeta()
        print(f"模型描述: {modelmeta.description}")
        print(f"模型领域: {modelmeta.domain}")
        print(f"模型版本: {modelmeta.version}")
        
        assert modelmeta is not None
