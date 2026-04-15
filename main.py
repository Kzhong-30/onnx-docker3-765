import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random
import os

class ONNXPrediction(object):     
    def __init__(self, onnx_model_path, image_path):     #my_model.onnx
        self.onnx_model_path = onnx_model_path
        self.ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.classes = ['ant', 'bee']
        self.image_path = image_path


    def predict(self):
        img = Image.open(self.image_path)
        img_tensor = self.transforms(img).unsqueeze(0)
        input_data = img_tensor.numpy()

        #Run inference
        outputs = self.ort_session.run(None, {'input': input_data})
        logits = outputs[0]

        # Apply softmax to convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits))  # subtract max for numerical stability
        probabilities = exp_logits / np.sum(exp_logits)

        # Get result
        classes = ['ant', 'bee']
        predicted_idx = np.argmax(probabilities)
        confidence = np.max(probabilities)

        print(f"Prediction: {classes[predicted_idx]}")
        print(f"Confidence: {confidence:.2%}")
        


if __name__ == "__main__":
    quantized_model_path = "my_model_quantized.onnx"

    #take a random image from the dataset
    img_class = random.choice(['ants', 'bees'])
    source_path = f"hymenoptera_data/val/{img_class}"
    all_files = [f for f in os.listdir(source_path) if f.endswith('.jpg')]
    #selected_file = random.choice(all_files)
    selected_files = random.sample(all_files, k=10)
    
    # Convert plural class name to singular for comparison
    ground_truth = img_class[:-1]  # 'ants' -> 'ant', 'bees' -> 'bee'
    
    for file in selected_files:
        image_path = f"{source_path}/{file}"
        print(image_path)
        print(f"Ground Truth: {ground_truth}")

        onnx_prediction = ONNXPrediction(quantized_model_path, image_path)
        onnx_prediction.predict()
        print("--------------------------------")

