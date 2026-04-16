import os
import pytest


class TestFrameworkValidation:
    def test_pytest_works(self):
        assert True, "Basic pytest assertion works"

    def test_test_directory_structure(self):
        assert os.path.exists("tests/"), "Tests directory exists"
        assert os.path.exists("tests/test_model_performance.py"), "Performance tests exist"
        assert os.path.exists("tests/test_api_integration.py"), "API tests exist"

    def test_model_file_exists(self):
        assert os.path.exists("my_model_quantized.onnx"), "Model file exists"

    def test_test_data_exists(self):
        assert os.path.exists("hymenoptera_data/val/ants"), "Validation data exists"
        assert os.path.exists("hymenoptera_data/val/bees"), "Validation data exists"

    def test_requirements_are_complete(self):
        assert os.path.exists("requirements.txt"), "Requirements file exists"
        with open("requirements.txt", "r") as f:
            content = f.read()
            assert "pytest" in content, "pytest in requirements"
            assert "pytest-html" in content, "pytest-html in requirements"
            assert "httpx" in content, "httpx in requirements"

    def test_docker_configuration_exists(self):
        assert os.path.exists("docker-compose.yml"), "docker-compose exists"
        assert os.path.exists("dockerfile"), "Dockerfile exists"
        assert os.path.exists("dockerfile.test"), "Test Dockerfile exists"

    def test_pytest_config_exists(self):
        assert os.path.exists("pytest.ini"), "pytest.ini exists"
        assert os.path.exists("run_tests.py"), "Test runner exists"
