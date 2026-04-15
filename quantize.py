from onnxruntime.quantization import quantize_dynamic, QuantType
import torchvision.transforms as transforms
import numpy as np
import os
import time
import onnxruntime as ort
from PIL import Image
import random
class Quantization(object):
    def __init__(self, model_path, quantized_model_path, image_path):
        self.model_path = model_path
        self.quantized_model_path = quantized_model_path
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.image_path = image_path

    def quantize_model(self):
        quantize_dynamic(self.model_path, 
                        self.quantized_model_path, 
                        weight_type=QuantType.QUInt8)

    def get_input_data(self):
        """Prepare input data from image"""
        img = Image.open(self.image_path)
        img_tensor = self.transforms(img).unsqueeze(0)
        return img_tensor.numpy()

    def predict(self, ort_session):
        input_data = self.get_input_data()

        # Run inference
        outputs = ort_session.run(None, {'input': input_data})
        logits = outputs[0]

        """
        was originally this: 
        prediction = outputs[0]
        predicted_idx = np.argmax(prediction)
        confidence = np.max(prediction) #<- but this outputs raw logit, leadingn to 139.29% probability
        So, you have to apply softmax to convert to probabilities.
        """

        # Apply softmax to convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits))  # subtract max for numerical stability
        probabilities = exp_logits / np.sum(exp_logits)

        # Get result
        classes = ['ant', 'bee']
        predicted_idx = np.argmax(probabilities)
        confidence = np.max(probabilities)

        print(f"Prediction: {classes[predicted_idx]}")
        print(f"Confidence: {confidence:.2%}")
        
        return probabilities

    def verify_quantization(self):
        # 1. File Size Check
        print("\n[1] FILE SIZE COMPARISON")
        original_size = os.path.getsize(self.model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(self.quantized_model_path) / (1024 * 1024)
        reduction = ((original_size - quantized_size) / original_size * 100)
        
        print(f"Original:  {original_size:.2f} MB")
        print(f"Quantized: {quantized_size:.2f} MB")
        print(f"Reduction: {reduction:.1f}%")

        # 2. Inference Speed Test
        print("\n[2] INFERENCE SPEED COMPARISON")
    
        input_data = self.get_input_data()
        
        # Original model
        sess_orig = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        start = time.time()
        for _ in range(50):
            _ = sess_orig.run(None, {'input': input_data})
        orig_time = time.time() - start
        
        # Quantized model
        sess_quant = ort.InferenceSession(self.quantized_model_path, providers=['CPUExecutionProvider'])
        start = time.time()
        for _ in range(50):
            _ = sess_quant.run(None, {'input': input_data})
        quant_time = time.time() - start
        
        speedup = orig_time / quant_time
        print(f"Original:  {orig_time:.3f}s (50 inferences)")
        print(f"Quantized: {quant_time:.3f}s (50 inferences)")
        print(f"Speedup:   {speedup:.2f}x")
        
        # 3. Accuracy Check
        print("\n[3] ACCURACY CHECK")
        print("Original model:")
        out_orig = self.predict(sess_orig)
        print("\nQuantized model:")
        out_quant = self.predict(sess_quant)

        max_diff = np.max(np.abs(out_orig - out_quant))
        mean_diff = np.mean(np.abs(out_orig - out_quant))
        
        print(f"\nProbability differences:")
        print(f"Max difference:  {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")

if __name__ == "__main__":
    model_path = "my_model.onnx"
    quantized_model_path = "my_model_quantized.onnx"

    #take a random image from the dataset
    img_class = random.choice(['ants', 'bees'])
    source_path = f"hymenoptera_data/val/{img_class}"
    all_files = [f for f in os.listdir(source_path) if f.endswith('.jpg')]
    selected_file = random.choice(all_files)
    image_path = f"{source_path}/{selected_file}"
    print(image_path)

    quantization = Quantization(model_path, quantized_model_path, image_path)
    quantization.quantize_model()
    quantization.verify_quantization()

