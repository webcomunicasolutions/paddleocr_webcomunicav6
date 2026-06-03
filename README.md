# PaddleOCR WebComunica v6 — Laboratorio de benchmark

Repo del **laboratorio v6**: prepara los contenedores de los experimentos OCR/VLM nuevos para desplegarlos en EasyPanel (GPU RTX 4070) y compararlos contra la producción actual (v5.6).

> Objetivo final: decidir con datos qué modelo sustituye (o no) al servidor de producción `paddleocr_webcomunicav5`. Objetivos: quitar dependencia de Gemini, mejorar precisión, reducir fallback a LLMWhisperer.

## Cómo se usa

1. Cada experimento es una carpeta con su `Dockerfile` (+ `docker-compose` / app).
2. **El despliegue lo hace el usuario en EasyPanel** apuntando a este repo (rama/carpeta), en la GPU del servidor gesfac.
3. Puertos de experimento: **9101-9104** (producción 8505 NO se toca). Coordinación de infra vía bridge con el proyecto `servidor_gesfac`.

## Experimentos (una carpeta por fase)

| Carpeta | Modelo | Qué prueba |
|---|---|---|
| `fase1_ppocrv5/` | PP-OCRv5 | re-test OCR en GPU (ya dio −3.4% en CPU) |
| `fase2_structurev3/` | PP-StructureV3 | tablas/layout → JSON |
| `fase3_paddleocr-vl/` | PaddleOCR-VL 1.6 (vLLM) | VLM end-to-end imagen→JSON ⭐ |
| `fase4_chatocr-ollama/` | PP-ChatOCRv4 + Ollama | KIE local (sin Gemini) |

## ⚠️ Importante
- **NUNCA** se suben facturas ni datos de clientes a este repo (ver `.gitignore`). El dataset y el ground truth viven solo en local.
- Sin credenciales ni `.env` en el repo.

Plan completo y hallazgos: en el proyecto local `paddleocr/` (`02_EXPERIMENTOS/PLAN.md`, `01_INVESTIGACION/HALLAZGOS_2026-06.md`, `HISTORIAL.md`).
