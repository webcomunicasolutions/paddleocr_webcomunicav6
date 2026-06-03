# Fase 2 — PP-StructureV3 (layout + tablas) en GPU
# Misma base que Fases 1 y 3 (CUDA 12.6). Pipeline PPStructureV3 de la librería paddleocr.
# La GPU se hereda del host (default-runtime: nvidia). Descarga modelos en 1er arranque.
FROM paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_XET=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils curl procps libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "numpy<2" \
    "paddleocr[doc-parser]>=3.4.0" \
    "Pillow>=10.0.0" \
    flask \
    waitress

WORKDIR /app
COPY app.py /app/

EXPOSE 8502
CMD ["python", "app.py"]
