#!/usr/bin/env python3
"""Fase 2 — Servicio PP-StructureV3 (layout + tablas) para benchmark de facturas.

Recibe un PDF/imagen y devuelve la transcripción estructurada (markdown + json del
layout, con tablas reconocidas). A diferencia del OCR plano, reconstruye la estructura
de tablas — útil para el objetivo futuro de extraer líneas de detalle de la factura.

Endpoints:
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
logger = logging.getLogger("fase2-structure")

app = Flask(__name__)
PORT = int(os.environ.get("FLASK_PORT", "8502"))
RENDER_DPI = int(os.environ.get("STRUCT_RENDER_DPI", "200"))
USE_TABLE = os.environ.get("STRUCT_USE_TABLE", "true").lower() == "true"
USE_FORMULA = os.environ.get("STRUCT_USE_FORMULA", "false").lower() == "true"
# Modelo de reconocimiento de texto. Para facturas en español conviene un modelo latino.
# Vacío = default del pipeline (chino-inglés). Ajustable por env si el español sale mal.
REC_MODEL = os.environ.get("STRUCT_REC_MODEL", "").strip()

pipeline = None
load_error = None


def init_pipeline():
    global pipeline, load_error
    try:
        logger.info("[INIT] Cargando PPStructureV3 (table=%s formula=%s rec=%s)...",
                    USE_TABLE, USE_FORMULA, REC_MODEL or "default")
        from paddleocr import PPStructureV3
        kwargs = {
            "device": "gpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            "use_table_recognition": USE_TABLE,
            "use_formula_recognition": USE_FORMULA,
        }
        if REC_MODEL:
            kwargs["text_recognition_model_name"] = REC_MODEL
        pipeline = PPStructureV3(**kwargs)
        logger.info("[INIT] PPStructureV3 listo.")
        return True
    except Exception as e:  # noqa: BLE001
        load_error = str(e)
        logger.exception("[INIT] Fallo cargando PPStructureV3: %s", e)
        return False


def pdf_to_pngs(src: str, workdir: str) -> list:
    ext = Path(src).suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        return [src]
    out_prefix = os.path.join(workdir, "page")
    subprocess.run(["pdftoppm", "-png", "-r", str(RENDER_DPI), src, out_prefix],
                   check=True, timeout=180)
    return sorted(str(p) for p in Path(workdir).glob("page*.png"))


def run_struct(png_path: str, workdir: str) -> dict:
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
        "struct_ready": pipeline is not None,
        "use_table": USE_TABLE,
        "use_formula": USE_FORMULA,
        "rec_model": REC_MODEL or "default",
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
                r = run_struct(png, page_dir)
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
