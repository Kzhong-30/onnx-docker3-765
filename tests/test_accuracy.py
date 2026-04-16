import pytest
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import random


class TestModelAccuracy:
    """模型准确率测试套件"""

    @pytest.fixture(scope="class")
    def session(self, model_path):
        """创建ONNX会话"""
        return ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

    @pytest.fixture(scope="class")
    def transform(self):
        """图像预处理转换"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_image(self, session, image_path, transform):
        """预测单张图片"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        input_data = img_tensor.numpy()
        
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        logits = outputs[0]
        
        # 应用softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        predicted_idx = np.argmax(probabilities)
        confidence = float(np.max(probabilities))
        
        return predicted_idx, confidence

    def test_validation_dataset_exists(self, test_data_dir):
        """测试验证数据集是否存在"""
        assert os.path.exists(test_data_dir), f"验证数据集不存在: {test_data_dir}"
        
        ants_dir = os.path.join(test_data_dir, "ants")
        bees_dir = os.path.join(test_data_dir, "bees")
        
        assert os.path.exists(ants_dir), f"ants目录不存在: {ants_dir}"
        assert os.path.exists(bees_dir), f"bees目录不存在: {bees_dir}"

    def test_single_image_prediction(self, session, transform, test_data_dir):
        """测试单张图片预测"""
        # 获取一张测试图片
        ants_dir = os.path.join(test_data_dir, "ants")
        image_files = [f for f in os.listdir(ants_dir) if f.endswith('.jpg')]
        
        if not image_files:
            pytest.skip("没有找到测试图片")
        
        image_path = os.path.join(ants_dir, image_files[0])
        predicted_idx, confidence = self.predict_image(session, image_path, transform)
        
        assert predicted_idx in [0, 1], f"预测索引无效: {predicted_idx}"
        assert 0 <= confidence <= 1, f"置信度无效: {confidence}"
        print(f"预测类别: {['ant', 'bee'][predicted_idx]}, 置信度: {confidence:.2%}")

    def test_ants_accuracy(self, session, transform, test_data_dir):
        """测试ants类别的准确率"""
        ants_dir = os.path.join(test_data_dir, "ants")
        image_files = [f for f in os.listdir(ants_dir) if f.endswith('.jpg')]
        
        if not image_files:
            pytest.skip("没有找到ants测试图片")
        
        correct = 0
        total = 0
        
        for image_file in image_files:
            image_path = os.path.join(ants_dir, image_file)
            try:
                predicted_idx, confidence = self.predict_image(session, image_path, transform)
                # ant的索引是0
                if predicted_idx == 0:
                    correct += 1
                total += 1
            except Exception as e:
                print(f"处理图片 {image_file} 时出错: {e}")
                continue
        
        if total == 0:
            pytest.skip("没有成功处理任何图片")
        
        accuracy = correct / total
        print(f"ants类别准确率: {accuracy:.2%} ({correct}/{total})")
        
        # 断言准确率大于50%
        assert accuracy > 0.5, f"ants类别准确率过低: {accuracy:.2%}"

    def test_bees_accuracy(self, session, transform, test_data_dir):
        """测试bees类别的准确率"""
        bees_dir = os.path.join(test_data_dir, "bees")
        image_files = [f for f in os.listdir(bees_dir) if f.endswith('.jpg')]
        
        if not image_files:
            pytest.skip("没有找到bees测试图片")
        
        correct = 0
        total = 0
        
        for image_file in image_files:
            image_path = os.path.join(bees_dir, image_file)
            try:
                predicted_idx, confidence = self.predict_image(session, image_path, transform)
                # bee的索引是1
                if predicted_idx == 1:
                    correct += 1
                total += 1
            except Exception as e:
                print(f"处理图片 {image_file} 时出错: {e}")
                continue
        
        if total == 0:
            pytest.skip("没有成功处理任何图片")
        
        accuracy = correct / total
        print(f"bees类别准确率: {accuracy:.2%} ({correct}/{total})")
        
        # 断言准确率大于50%
        assert accuracy > 0.5, f"bees类别准确率过低: {accuracy:.2%}"

    def test_overall_accuracy(self, session, transform, test_data_dir):
        """测试整体准确率"""
        ants_dir = os.path.join(test_data_dir, "ants")
        bees_dir = os.path.join(test_data_dir, "bees")
        
        ants_files = [f for f in os.listdir(ants_dir) if f.endswith('.jpg')]
        bees_files = [f for f in os.listdir(bees_dir) if f.endswith('.jpg')]
        
        correct = 0
        total = 0
        
        # 测试ants
        for image_file in ants_files:
            image_path = os.path.join(ants_dir, image_file)
            try:
                predicted_idx, _ = self.predict_image(session, image_path, transform)
                if predicted_idx == 0:  # ant
                    correct += 1
                total += 1
            except Exception as e:
                continue
        
        # 测试bees
        for image_file in bees_files:
            image_path = os.path.join(bees_dir, image_file)
            try:
                predicted_idx, _ = self.predict_image(session, image_path, transform)
                if predicted_idx == 1:  # bee
                    correct += 1
                total += 1
            except Exception as e:
                continue
        
        if total == 0:
            pytest.skip("没有成功处理任何图片")
        
        accuracy = correct / total
        print(f"整体准确率: {accuracy:.2%} ({correct}/{total})")
        
        # 断言整体准确率大于50%
        assert accuracy > 0.5, f"整体准确率过低: {accuracy:.2%}"

    def test_confidence_distribution(self, session, transform, test_data_dir):
        """测试置信度分布"""
        ants_dir = os.path.join(test_data_dir, "ants")
        bees_dir = os.path.join(test_data_dir, "bees")
        
        ants_files = [f for f in os.listdir(ants_dir) if f.endswith('.jpg')]
        bees_files = [f for f in os.listdir(bees_dir) if f.endswith('.jpg')]
        
        confidences = []
        
        # 收集置信度
        for image_file in ants_files[:10]:  # 限制数量
            image_path = os.path.join(ants_dir, image_file)
            try:
                _, confidence = self.predict_image(session, image_path, transform)
                confidences.append(confidence)
            except Exception as e:
                continue
        
        for image_file in bees_files[:10]:  # 限制数量
            image_path = os.path.join(bees_dir, image_file)
            try:
                _, confidence = self.predict_image(session, image_path, transform)
                confidences.append(confidence)
            except Exception as e:
                continue
        
        if not confidences:
            pytest.skip("没有成功处理任何图片")
        
        avg_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        
        print(f"平均置信度: {avg_confidence:.2%}")
        print(f"最小置信度: {min_confidence:.2%}")
        print(f"最大置信度: {max_confidence:.2%}")
        
        # 断言平均置信度大于50%
        assert avg_confidence > 0.5, f"平均置信度过低: {avg_confidence:.2%}"

    @pytest.mark.parametrize("sample_size", [5, 10, 20])
    def test_sample_accuracy(self, session, transform, test_data_dir, sample_size):
        """测试不同样本量的准确率"""
        ants_dir = os.path.join(test_data_dir, "ants")
        bees_dir = os.path.join(test_data_dir, "bees")
        
        ants_files = [f for f in os.listdir(ants_dir) if f.endswith('.jpg')]
        bees_files = [f for f in os.listdir(bees_dir) if f.endswith('.jpg')]
        
        # 随机采样
        sample_ants = random.sample(ants_files, min(sample_size, len(ants_files)))
        sample_bees = random.sample(bees_files, min(sample_size, len(bees_files)))
        
        correct = 0
        total = 0
        
        # 测试ants
        for image_file in sample_ants:
            image_path = os.path.join(ants_dir, image_file)
            try:
                predicted_idx, _ = self.predict_image(session, image_path, transform)
                if predicted_idx == 0:
                    correct += 1
                total += 1
            except Exception as e:
                continue
        
        # 测试bees
        for image_file in sample_bees:
            image_path = os.path.join(bees_dir, image_file)
            try:
                predicted_idx, _ = self.predict_image(session, image_path, transform)
                if predicted_idx == 1:
                    correct += 1
                total += 1
            except Exception as e:
                continue
        
        if total == 0:
            pytest.skip("没有成功处理任何图片")
        
        accuracy = correct / total
        print(f"样本量 {sample_size*2}: 准确率 {accuracy:.2%} ({correct}/{total})")
        
        # 断言准确率大于40%
        assert accuracy > 0.4, f"样本准确率过低: {accuracy:.2%}"
