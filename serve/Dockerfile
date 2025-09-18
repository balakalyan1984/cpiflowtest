FROM python:3.12-slim
ENV LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app/src
COPY requirements.txt /app/src/requirements.txt
RUN pip install --no-cache-dir -r /app/src/requirements.txt
COPY serve_cpiflow_kserve.py /app/src/serve_cpiflow_kserve.py
RUN chmod -R 0777 /app
EXPOSE 9001
CMD ["uvicorn","serve_cpiflow_kserve:app","--host","0.0.0.0","--port","9001"]
