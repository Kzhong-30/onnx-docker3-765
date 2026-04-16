import pytest
import numpy as np
from PIL import Image
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import preprocess_image


class TestAccuracy:
    
    MIN_ACCURACY_THRESHOLD = 0.70
    MIN_CONFIDENCE_THRESHOLD = 0.50
    
    def test_prediction_output_format(self, onnx_session, sample_image_tensor, classes):
        input_name = onnx_session.get_inputs()[0].name
        outputs = onnx_session.run(None, {input_name: sample_image_tensor})
        
        assert len(outputs) > 0, "No output from model"
        assert outputs[0].shape[0] == 1, "Batch size should be 1"
        assert outputs[0].shape[1] == 2, "Should have 2 class outputs"
    
    def test_prediction_probabilities_valid(self, onnx_session, sample_image_tensor):
        input_name = onnx_session.get_inputs()[0].name
        outputs = onnx_session.run(None, {input_name: sample_image_tensor})
        logits = outputs[0]
        
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        assert np.all(probabilities >= 0), "Probabilities should be non-negative"
        assert np.all(probabilities <= 1), "Probabilities should be <= 1"
        assert np.isclose(np.sum(probabilities), 1.0), "Probabilities should sum to 1"
    
    def test_single_image_prediction(self, onnx_session, test_images, classes):
        if not test_images:
            pytest.skip("No test images available")
        
        test_img = test_images[0]
        img_tensor = preprocess_image(test_img['path'])
        
        input_name = onnx_session.get_inputs()[0].name
        outputs = onnx_session.run(None, {input_name: img_tensor})
        
        logits = outputs[0]
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        predicted_idx = np.argmax(probabilities)
        predicted_class = classes[predicted_idx]
        
        assert predicted_class in classes, f"Predicted class {predicted_class} not in valid classes"
    
    def test_accuracy_on_validation_set(self, onnx_session, test_images, classes):
        if not test_images:
            pytest.skip("No test images available")
        
        correct_predictions = 0
        total_predictions = 0
        
        for test_img in test_images:
            try:
                img_tensor = preprocess_image(test_img['path'])
                
                input_name = onnx_session.get_inputs()[0].name
                outputs = onnx_session.run(None, {input_name: img_tensor})
                
                logits = outputs[0]
                exp_logits = np.exp(logits - np.max(logits))
                probabilities = exp_logits / np.sum(exp_logits)
                
                predicted_idx = np.argmax(probabilities)
                predicted_class = classes[predicted_idx]
                
                if predicted_class == test_img['label']:
                    correct_predictions += 1
                total_predictions += 1
            except Exception as e:
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        assert accuracy >= self.MIN_ACCURACY_THRESHOLD, \
            f"Accuracy {accuracy:.2%} is below threshold {self.MIN_ACCURACY_THRESHOLD:.2%}"
    
    def test_per_class_accuracy(self, onnx_session, test_images, classes):
        if not test_images:
            pytest.skip("No test images available")
        
        class_correct = {'ant': 0, 'bee': 0}
        class_total = {'ant': 0, 'bee': 0}
        
        for test_img in test_images:
            try:
                img_tensor = preprocess_image(test_img['path'])
                
                input_name = onnx_session.get_inputs()[0].name
                outputs = onnx_session.run(None, {input_name: img_tensor})
                
                logits = outputs[0]
                exp_logits = np.exp(logits - np.max(logits))
                probabilities = exp_logits / np.sum(exp_logits)
                
                predicted_idx = np.argmax(probabilities)
                predicted_class = classes[predicted_idx]
                
                class_total[test_img['label']] += 1
                if predicted_class == test_img['label']:
                    class_correct[test_img['label']] += 1
            except Exception:
                continue
        
        for class_name in ['ant', 'bee']:
            if class_total[class_name] > 0:
                class_accuracy = class_correct[class_name] / class_total[class_name]
                assert class_accuracy >= 0.5, \
                    f"Class '{class_name}' accuracy {class_accuracy:.2%} is too low"
    
    def test_confidence_scores(self, onnx_session, test_images):
        if not test_images:
            pytest.skip("No test images available")
        
        low_confidence_count = 0
        total_predictions = 0
        
        for test_img in test_images:
            try:
                img_tensor = preprocess_image(test_img['path'])
                
                input_name = onnx_session.get_inputs()[0].name
                outputs = onnx_session.run(None, {input_name: img_tensor})
                
                logits = outputs[0]
                exp_logits = np.exp(logits - np.max(logits))
                probabilities = exp_logits / np.sum(exp_logits)
                
                confidence = np.max(probabilities)
                if confidence < self.MIN_CONFIDENCE_THRESHOLD:
                    low_confidence_count += 1
                total_predictions += 1
            except Exception:
                continue
        
        low_confidence_ratio = low_confidence_count / total_predictions if total_predictions > 0 else 0
        assert low_confidence_ratio < 0.3, \
            f"Too many low confidence predictions: {low_confidence_ratio:.2%}"
    
    def test_confusion_matrix(self, onnx_session, test_images, classes):
        if not test_images:
            pytest.skip("No test images available")
        
        confusion_matrix = {
            'ant': {'ant': 0, 'bee': 0},
            'bee': {'ant': 0, 'bee': 0}
        }
        
        for test_img in test_images:
            try:
                img_tensor = preprocess_image(test_img['path'])
                
                input_name = onnx_session.get_inputs()[0].name
                outputs = onnx_session.run(None, {input_name: img_tensor})
                
                logits = outputs[0]
                exp_logits = np.exp(logits - np.max(logits))
                probabilities = exp_logits / np.sum(exp_logits)
                
                predicted_idx = np.argmax(probabilities)
                predicted_class = classes[predicted_idx]
                
                confusion_matrix[test_img['label']][predicted_class] += 1
            except Exception:
                continue
        
        for true_class in ['ant', 'bee']:
            total = sum(confusion_matrix[true_class].values())
            if total > 0:
                correct = confusion_matrix[true_class][true_class]
                accuracy = correct / total
                assert accuracy >= 0.5, \
                    f"Class '{true_class}' has low accuracy: {accuracy:.2%}"
    
    def test_prediction_consistency(self, onnx_session, test_images, classes):
        if not test_images:
            pytest.skip("No test images available")
        
        test_img = test_images[0]
        img_tensor = preprocess_image(test_img['path'])
        
        input_name = onnx_session.get_inputs()[0].name
        predictions = []
        
        for _ in range(10):
            outputs = onnx_session.run(None, {input_name: img_tensor})
            logits = outputs[0]
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            predicted_idx = np.argmax(probabilities)
            predictions.append(predicted_idx)
        
        assert len(set(predictions)) == 1, "Predictions are not consistent across runs"
    
    def test_output_logits_reasonable(self, onnx_session, sample_image_tensor):
        input_name = onnx_session.get_inputs()[0].name
        outputs = onnx_session.run(None, {input_name: sample_image_tensor})
        logits = outputs[0]
        
        assert not np.isnan(logits).any(), "Logits contain NaN values"
        assert not np.isinf(logits).any(), "Logits contain infinite values"
        assert np.abs(logits).max() < 1000, "Logits values are too large"
    
    def test_model_comparison_quantized_vs_original(self, onnx_session, original_onnx_session, 
                                                     test_images, classes):
        if original_onnx_session is None:
            pytest.skip("Original model not available")
        
        if not test_images:
            pytest.skip("No test images available")
        
        test_img = test_images[0]
        img_tensor = preprocess_image(test_img['path'])
        
        quantized_input_name = onnx_session.get_inputs()[0].name
        quantized_outputs = onnx_session.run(None, {quantized_input_name: img_tensor})
        quantized_logits = quantized_outputs[0]
        
        original_input_name = original_onnx_session.get_inputs()[0].name
        original_outputs = original_onnx_session.run(None, {original_input_name: img_tensor})
        original_logits = original_outputs[0]
        
        quantized_pred = np.argmax(quantized_logits)
        original_pred = np.argmax(original_logits)
        
        assert quantized_pred == original_pred, \
            f"Quantized and original models predict different classes: {quantized_pred} vs {original_pred}"
