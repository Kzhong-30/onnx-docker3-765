# Super Simple CNN Transfer Learning with ONNX Conversion

## Overview

This project demonstrates a complete machine learning workflow from training a CNN using transfer learning to deploying a quantized ONNX model for efficient CPU inference.

<p align=center>
<img src=./diagram.png>
<img src=./bee.png><img src=./graph.png height=300>
</p>

## Goals

The primary goals of this project are to:

1. **Leverage GPU acceleration** - Use Google Colab Pro A100 GPU Runtime to quickly train a fine-tuned CNN. Note: I purchased the Google Colab Pro subscription which provides access to 80GB VRAM. If you do not have a subscription, the training will take longer on the free Tesla 4.
2. **Transfer learning** - Fine-tune a pre-trained model using the classic PyTorch transfer learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
3. **Model conversion** - Convert the trained PyTorch model to ONNX format for cross-platform deployment
4. **Model optimization** - Apply INT8 quantization to reduce model size and improve inference speed
5. **Local deployment** - Run the optimized model locally on CPU for inference

## Dataset

The project uses the classic **Hymenoptera** dataset (ants and bees) from the PyTorch tutorial. Make sure to download the `hymenoptera_data.zip` file before training.

## Workflow

### 1. Training (Google Colab)

The Jupyter notebook follows the PyTorch transfer learning tutorial to:
- Load and preprocess the hymenoptera dataset
- Fine-tune a pre-trained CNN (ResNet/VGG) on the ants vs bees classification task
- Leverage A100 GPU for fast training
- Convert the trained PyTorch model to ONNX format
- Download the ONNX model from Google Drive to your local machine

**Note**: ONNX conversion makes the runtime faster and platform-agnostic, but does not inherently reduce model size. That's where quantization comes in.

### 2. Quantization (`quantize.py`)

The quantization script performs INT8 quantization on the ONNX model to significantly reduce file size and improve inference performance.

**Purpose**: 
- Quantize the ONNX model weights from FP32 to UINT8
- Verify the quantized model through comprehensive testing
- Compare file size, inference speed, and accuracy between original and quantized models

**How to run**:
```bash
python quantize.py
```

**Sample Output**:
```
hymenoptera_data/val/bees/1799729694_0c40101071.jpg

[1] FILE SIZE COMPARISON
Original:  42.63 MB
Quantized: 10.71 MB
Reduction: 74.9%

[2] INFERENCE SPEED COMPARISON
Original:  0.587s (50 inferences)
Quantized: 0.888s (50 inferences)
Speedup:   0.66x **slowdown**, see below for why!

[3] ACCURACY CHECK
Original model:
Prediction: bee
Confidence: 94.50%

Quantized model:
Prediction: bee
Confidence: 96.21%

Probability differences:
Max difference:  0.017028
Mean difference: 0.017028
```

**Key Features**:
- Randomly selects a validation image from ants or bees directory
- Quantizes model using dynamic quantization with UINT8 weights
- Provides comprehensive verification:
  - File size reduction (typically 70-75%)
  - Inference speed comparison
  - Accuracy/confidence comparison between original and quantized models
  - Probability difference metrics

** Performance Note for Mac Users**:
If running on macOS, you may observe **slower inference** with the quantized model (0.66x slowdown in the example above). This is expected because:
- The current implementation uses `CPUExecutionProvider` which doesn't have optimized INT8 operations on Mac
- Dynamic quantization adds dequantization overhead that can exceed computational savings
- **To achieve speedup on Mac**: Use ONNX Runtime with CoreML or Metal execution providers
- **Alternative**: Consider using static quantization (quantizes both weights AND activations) instead of dynamic quantization

Despite slower inference, quantization still provides significant benefits:
- 70-75% file size reduction
- Lower memory footprint
- Nearly identical accuracy

### 3. Inference (`main.py`)

The main inference script runs the quantized ONNX model locally on CPU for real-world predictions.

**Purpose**:
- Load the quantized ONNX model
- Perform batch inference on validation images using CPU
- Classify images as ant or bee with confidence scores
- Compare predictions against ground truth labels

**How to run**:
```bash
python main.py
```

**Features**:
- CPU-optimized inference using ONNX Runtime
- Image preprocessing with standard transformations (resize, center crop, normalization)
- Softmax probability conversion for proper confidence scores
- Randomly samples 10 images from validation set (either all ants or all bees)
- Displays ground truth for easy accuracy verification

**Sample Output**:
```
hymenoptera_data/val/ants/1440002809_b268d9a66a.jpg
Ground Truth: ant
Prediction: ant
Confidence: 81.07%
--------------------------------
hymenoptera_data/val/ants/2219621907_47bc7cc6b0.jpg
Ground Truth: ant
Prediction: ant
Confidence: 97.63%
--------------------------------
hymenoptera_data/val/ants/470127071_8b8ee2bd74.jpg
Ground Truth: ant
Prediction: bee
Confidence: 79.15%
--------------------------------
hymenoptera_data/val/ants/8124241_36b290d372.jpg
Ground Truth: ant
Prediction: ant
Confidence: 86.21%
--------------------------------
```

**Key Observations**:
- Most predictions are correct with high confidence (80-97%)
- Occasional misclassifications occur (e.g., ant predicted as bee)
- The model's confidence scores provide insight into prediction certainty
- Lower confidence scores may indicate more challenging images

## Docker Deployment with FastAPI

This section follows the [FastAPI Docker deployment guide](https://fastapi.tiangolo.com/deployment/docker/#create-the-fastapi-code).

### FastAPI Application (`onnx_app.py`)

The FastAPI application serves the quantized ONNX model via REST API.

**Endpoints:**
- `GET /` - Health check endpoint
- `POST /predict` - Upload an image file to get ant/bee prediction

**Features:**
- Accepts image uploads via multipart/form-data
- Returns prediction with confidence scores and probabilities for both classes
- Runs on ONNX Runtime for efficient CPU inference

### Build Docker Image

The Dockerfile is optimized following FastAPI best practices:
- Installs dependencies first (for Docker layer caching)
- Copies application code and model
- Includes validation data for testing
- Uses Uvicorn as the ASGI server

**Build the image:**
```bash
docker build -t myimage .
```

### Run Docker Container Locally

**Start the container:**
```bash
docker run -d --name onnxmodel -p 80:80 myimage
```

**Flags explained:**
- `-d` - Run in detached mode (background)
- `--name onnxmodel` - Name the container for easy reference
- `-p 80:80` - Map local port 80 to container port 80
- `myimage` - Use the image we built

**Access the API:**
- Root endpoint: `http://localhost/`
- Interactive API docs: `http://localhost/docs`
- Alternative docs: `http://localhost/redoc`

**Test with a validation image using curl:**
```bash
curl -X POST "http://localhost/predict" \
  -F "file=@hymenoptera_data/val/ants/10308379_1b6c72e180.jpg"
```

**Expected response:**
```json
{
  "prediction": "ant",
  "confidence": "97.63%",
  "confidence_score": 0.9763,
  "probabilities": {
    "ant": 0.9763,
    "bee": 0.0237
  }
}
```

### Push to Docker Hub

To share your image on Docker Hub:

```bash
# Tag the image with your Docker Hub username
docker tag myimage your-dockerhub-username/onnx-fastapi:latest

# Login to Docker Hub
docker login

# Push the image
docker push your-dockerhub-username/onnx-fastapi:latest
```

**Others can now pull and run your image:**

For example: 

```bash
docker pull cordun/onnx-fastapi:latest
docker run -d --name onnxmodel -p 80:80 cordun/onnx-fastapi:latest
```

### Useful Docker Commands

```bash
# View running containers
docker ps

# View container logs
docker logs onnxmodel

# Stop the container
docker stop onnxmodel

# Remove the container
docker rm onnxmodel

# Remove the image
docker rmi myimage
```



## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- PyTorch
- ONNX Runtime
- torchvision
- Pillow
- NumPy
- python-multipart (for file uploads)

## Project Structure

```
.
├── README.md
├── dockerfile                 # Docker image configuration
├── requirements.txt           # Python dependencies
├── onnx_app.py               # FastAPI application for serving model
├── main.py                    # Local inference script
├── quantize.py                # Quantization and verification script
├── hymenoptera_data/          # Dataset directory
│   ├── train/
│   │   ├── ants/
│   │   └── bees/
│   └── val/
│       ├── ants/
│       └── bees/
├── my_model.onnx              # Original ONNX model (from Colab)
└── my_model_quantized.onnx    # Quantized model (generated locally)
```

## Notes

- The quantization process uses dynamic quantization, which quantizes weights to UINT8 while keeping activations in FP32
- File size reduction is significant (70-75%), making the model much more portable
- **Inference speed varies by platform**: 
  - **Mac users**: Will likely see slowdowns with CPU-based dynamic quantization. Use CoreML/Metal providers or static quantization for speedup
  - **Linux/Windows with Intel AVX512_VNNI or ARM with INT8 support**: May see actual speedups
  - GPU inference with proper quantization support provides the best speedup
- Accuracy degradation is minimal (typically < 2% difference in probabilities)

