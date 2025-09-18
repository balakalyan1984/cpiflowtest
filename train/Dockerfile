FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN mkdir -p /app/src /app/data /app/model
COPY requirements.txt /app/src/requirements.txt
RUN pip install --no-cache-dir -r /app/src/requirements.txt
COPY train_cpiflow_pipeline.py /app/src/train_cpiflow_pipeline.py
RUN chmod -R 0777 /app
CMD ["python","/app/src/train_cpiflow_pipeline.py"]
