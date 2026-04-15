# build a docker image for FastAPI

FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./onnx_app.py /code/onnx_app.py
COPY ./my_model_quantized.onnx /code/my_model_quantized.onnx

# Copy validation data for testing
COPY ./hymenoptera_data/val /code/hymenoptera_data/val

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "onnx_app:app", "--host", "0.0.0.0", "--port", "80"]