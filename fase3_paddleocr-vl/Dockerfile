# Fase 3 — PaddleOCR-VL (VLM end-to-end) en GPU
# Despliegue NATIVO (sin vLLM): pipeline PaddleOCRVL de la librería paddleocr.
# Misma imagen base que Fase 1 (CUDA 12.6) -> paddlepaddle-gpu 3.2.1 cu126 encaja.
# La GPU se hereda del host (default-runtime: nvidia). VRAM del modelo ~2-3 GB.
FROM paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# El CDN "xet" de HuggingFace se cuelga desde este servidor -> descarga HTTPS normal (gotcha Fase 1)
ENV HF_HUB_DISABLE_XET=1

# Dependencias de sistema: PDF->imagen y libs de OpenCV/visión
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils curl procps libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# La imagen base trae paddlepaddle-gpu + CUDA + cudnn. Fijamos la versión que pide
# la model card de PaddleOCR-VL (3.2.1 cu126) por si la base trae otra, y el pipeline VL.
RUN pip install --no-cache-dir \
    "paddlepaddle-gpu==3.2.1" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ || true
RUN pip install --no-cache-dir \
    "numpy<2" \
    "paddleocr[doc-parser]>=3.4.0" \
    pdf2image \
    "Pillow>=10.0.0" \
    flask \
    waitress

WORKDIR /app
COPY app.py /app/

EXPOSE 8504
CMD ["python", "app.py"]
