#!/usr/bin/env python3
"""Fase 3 — Servicio PaddleOCR-VL (VLM end-to-end) para benchmark de facturas.

Recibe un PDF/imagen y devuelve la transcripción ESTRUCTURADA (markdown + json del
layout) que produce el pipeline PaddleOCRVL. A diferencia de PP-OCRv3/v5 (texto plano
desordenado), aquí el texto sale ordenado por bloques/tablas, lo que debería permitir
extraer campos de factura de forma fiable (objetivo: quitar Gemini).

Despliegue NATIVO (sin vLLM). El modelo (~2-3 GB VRAM) se descarga en el primer arranque
(por eso HF_HUB_DISABLE_XET=1 es imprescindible). Endpoints:
  GET  /health              -> estado del pipeline
  POST /process (file=...)  -> {markdown, json, stats}
"""
import os
import time
import json
import logging
import tempfile
import subprocess
from pathlib import Path

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from waitress import serve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fase3-vl")

app = Flask(__name__)
PORT = int(os.environ.get("FLASK_PORT", "8504"))
PIPELINE_VERSION = os.environ.get("VL_PIPELINE_VERSION", "v1")
RENDER_DPI = int(os.environ.get("VL_RENDER_DPI", "200"))

pipeline = None
load_error = None


def init_pipeline():
    """Carga el pipeline PaddleOCRVL una sola vez (descarga modelo en 1er arranque)."""
    global pipeline, load_error
    try:
        logger.info("[INIT] Cargando PaddleOCRVL (pipeline_version=%s)...", PIPELINE_VERSION)
        from paddleocr import PaddleOCRVL
        pipeline = PaddleOCRVL(pipeline_version=PIPELINE_VERSION)
        logger.info("[INIT] PaddleOCRVL listo.")
        return True
    except Exception as e:  # noqa: BLE001
        load_error = str(e)
        logger.exception("[INIT] Fallo cargando PaddleOCRVL: %s", e)
        return False


def pdf_to_pngs(src: str, workdir: str) -> list:
    """Convierte un PDF a una lista de PNG (una por página) con pdftoppm. Si ya es
    imagen, la devuelve tal cual."""
    ext = Path(src).suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        return [src]
    out_prefix = os.path.join(workdir, "page")
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(RENDER_DPI), src, out_prefix],
        check=True, timeout=180,
    )
    return sorted(Path(workdir).glob("page*.png").__iter__(), key=lambda p: p.name)


def run_vl(png_path: str, workdir: str) -> dict:
    """Corre el pipeline sobre UNA imagen y devuelve {markdown, json} leyendo los
    artefactos que escribe save_to_* (robusto frente a nombres de salida)."""
    md_dir = Path(workdir) / "md"
    js_dir = Path(workdir) / "js"
    md_dir.mkdir(exist_ok=True)
    js_dir.mkdir(exist_ok=True)
    output = pipeline.predict(str(png_path))
    for res in output:
        try:
            res.save_to_markdown(save_path=str(md_dir))
        except Exception as e:  # noqa: BLE001
            logger.warning("save_to_markdown falló: %s", e)
        try:
            res.save_to_json(save_path=str(js_dir))
        except Exception as e:  # noqa: BLE001
            logger.warning("save_to_json falló: %s", e)
    md = "\n\n".join(p.read_text(encoding="utf-8", errors="ignore")
                     for p in sorted(md_dir.glob("**/*.md")))
    js = []
    for p in sorted(js_dir.glob("**/*.json")):
        try:
            js.append(json.loads(p.read_text(encoding="utf-8", errors="ignore")))
        except Exception:  # noqa: BLE001
            pass
    return {"markdown": md, "json": js}


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy" if pipeline is not None else "loading",
        "vl_ready": pipeline is not None,
        "pipeline_version": PIPELINE_VERSION,
        "load_error": load_error,
    })


@app.route("/process", methods=["POST"])
def process():
    if pipeline is None:
        return jsonify({"success": False, "error": "pipeline not ready", "load_error": load_error}), 503
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"success": False, "error": "no file"}), 400

    f = request.files["file"]
    ext = Path(f.filename).suffix.lower()
    if ext not in (".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        return jsonify({"success": False, "error": f"unsupported: {ext}"}), 400

    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, secure_filename(f.filename) or ("in" + ext))
        f.save(src)
        try:
            pngs = pdf_to_pngs(src, td)
            md_all, json_all = [], []
            for i, png in enumerate(pngs):
                page_dir = os.path.join(td, f"out{i}")
                os.makedirs(page_dir, exist_ok=True)
                r = run_vl(png, page_dir)
                md_all.append(r["markdown"])
                json_all.extend(r["json"])
            elapsed = round(time.time() - t0, 3)
            return jsonify({
                "success": True,
                "markdown": "\n\n---\n\n".join(md_all),
                "json": json_all,
                "stats": {"pages": len(pngs), "processing_time": elapsed},
            })
        except subprocess.CalledProcessError as e:
            return jsonify({"success": False, "error": f"pdf->png falló: {e}"}), 500
        except Exception as e:  # noqa: BLE001
            logger.exception("[PROCESS] error")
            return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    init_pipeline()
    logger.info("[START] Sirviendo en :%d (waitress)", PORT)
    serve(app, host="0.0.0.0", port=PORT, threads=2)
