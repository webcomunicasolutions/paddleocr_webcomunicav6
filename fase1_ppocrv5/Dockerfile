# Fase 1 — PP-OCRv5 en GPU
# Base oficial PaddlePaddle GPU (CUDA 12.6; el servidor tiene driver 580/CUDA 13, compatible).
# La GPU se hereda automáticamente del host (default-runtime: nvidia en daemon.json).
FROM paddlepaddle/paddle:3.3.0-gpu-cuda12.6-cudnn9.5

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Dependencias de sistema (PDF/imagen) que usa app.py
RUN apt-get update && apt-get install -y --no-install-recommends \
    jq qpdf poppler-utils imagemagick ghostscript pdftk curl bc procps libgomp1 \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && (sed -i '/<policy domain="coder" rights="none" pattern="PDF"/s/rights="none"/rights="read|write"/' /etc/ImageMagick-6/policy.xml || true)

# La imagen base YA trae paddlepaddle-gpu + CUDA + cudnn.
# Solo añadimos PaddleOCR y las libs de la app. numpy<2 (paddleocr rompe con NumPy 2.x).
RUN pip install --no-cache-dir \
    "numpy<2" \
    opencv-python \
    paddleocr \
    pdf2image==1.16.3 \
    reportlab==4.0.4 \
    pdfplumber \
    "Pillow>=10.0.0" \
    PyPDF2 \
    PyMuPDF \
    flask \
    waitress

WORKDIR /app
COPY app.py /app/

EXPOSE 8503
CMD ["python", "app.py"]
