FROM python:3.9-slim
RUN pip install --no-cache-dir numpy fastapi python-multipart uvicorn onnxruntime Pillow opencv-python-headless
WORKDIR /app
ADD . /app
CMD ["uvicorn", "service:api", "--host", "0.0.0.0", "--port", "80"]
