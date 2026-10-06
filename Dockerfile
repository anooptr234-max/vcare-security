FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY loitering_detector.py pipeline.py store.py report.py .

ENTRYPOINT ["python", "pipeline.py"]
CMD ["--source", "rtsp://camera:554/stream", "--db", "/data/security.db",
     "--dwell", "300"]
