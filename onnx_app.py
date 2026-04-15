# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import io

app = FastAPI()

# Load the quantized ONNX model
onnx_model_path = "my_model_quantized.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# Image preprocessing transforms
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels
classes = ['ant', 'bee']

@app.get("/")
def read_root():
    return {"message": "ONNX Model Serving API is live"}
    
# most of the /predict endpoint is from main.py
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess the image
        img_tensor = image_transforms(img).unsqueeze(0)
        input_data = img_tensor.numpy()
        
        # Run inference
        outputs = ort_session.run(None, {'input': input_data})
        logits = outputs[0]
        
        # Apply softmax to convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Get prediction
        predicted_idx = np.argmax(probabilities)
        confidence = float(np.max(probabilities))
        
        return {
            "prediction": classes[predicted_idx],
            "confidence": f"{confidence:.2%}",
            "confidence_score": confidence,
            "probabilities": {
                "ant": float(probabilities[0][0]),
                "bee": float(probabilities[0][1])
            }
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
