# Fase 2 — PP-StructureV3 (layout + tablas)

Pipeline `PPStructureV3` de la librería `paddleocr`: OCR + detección de layout +
**reconocimiento de tablas** → markdown/JSON estructurado. Mismo patrón de despliegue
que Fases 1 y 3 (imagen base CUDA 12.6, GPU heredada del host).

## Para qué sirve (y para qué no)
- El OCR plano (v3/v5) ya captura los campos de **cabecera** (CIF, nº, total) al 96-98%
  → PP-StructureV3 NO es la vía para quitar Gemini de la cabecera.
- Su valor está en reconstruir las **tablas de detalle** → relevante para el objetivo
  futuro de extraer las **líneas** de cada factura (descripción, cantidad, precio).

## Despliegue en EasyPanel (igual que Fases 1 y 3)
1. App desde el repo, carpeta `fase2_structurev3`. Puerto host **9102** → 8502.
2. Pegar A MANO en *Environment*:
   - `HF_HUB_DISABLE_XET=1`  ← **crítico**
   - `FLASK_PORT=8502`
   - `STRUCT_RENDER_DPI=200`
   - `STRUCT_USE_TABLE=true`
   - `STRUCT_USE_FORMULA=false`
   - (opcional) `STRUCT_REC_MODEL=latin_PP-OCRv5_mobile_rec` si el español sale mal
3. 1er arranque: descarga modelos de layout + tabla (varios GB).
4. Verificar: `sudo docker exec <cid> curl -s localhost:8502/health` → `struct_ready:true`.

## Endpoints
- `GET /health` → `{status, struct_ready, use_table, load_error}`
- `POST /process` (multipart `file=@factura.pdf`) → `{success, markdown, json, stats}`

## Benchmark
Mismo patrón que Fases 1/3 (runner que llama a `/process`, guarda markdown+json).
Métrica de Fase 2: presencia de campos en el markdown (reutilizar `../../score_fase1.py`)
**y** calidad de las **tablas** reconocidas (¿totales/bases/IVA bien estructurados?,
¿líneas de detalle completas?). Es la base de evaluación para los line items.

## Notas / riesgos
- El modelo de texto por defecto de PP-StructureV3 es chino-inglés; para español puede
  necesitar el modelo latino (`STRUCT_REC_MODEL`). Se ajusta tras el primer arranque.
- La API de `save_to_*` se lee desde artefactos (igual que Fase 3) para robustez.
