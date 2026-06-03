# Fase 3 — PaddleOCR-VL (VLM end-to-end) ⭐

El candidato que SÍ puede sustituir a Gemini. Modelo `PaddlePaddle/PaddleOCR-VL`
(~0.9B params, Apache 2.0), servido **nativo** con el pipeline `PaddleOCRVL` de la
librería `paddleocr` (sin vLLM). Imagen → markdown/JSON **estructurado y ordenado**.

## Por qué nativo y no vLLM
- `paddlepaddle-gpu==3.2.1` (CUDA 12.6) encaja con la imagen base de Fase 1 (ya validada
  en este servidor). vLLM exigiría CUDA 12.9 + nightly + orquestar server+cliente.
- VRAM ~2-3 GB → **convive con `ia_ollama`** en la 4070 (vLLM al 90% causaría OOM).
- Para 21 facturas el throughput de vLLM no aporta nada.

## Despliegue en EasyPanel (igual que Fase 1)
1. Crear App desde el repo, carpeta `fase3_paddleocr-vl`. Puerto host **9103** → 8504.
2. **Pegar A MANO** en la pestaña *Environment* (EasyPanel NO lee el compose):
   - `HF_HUB_DISABLE_XET=1`  ← **crítico**, sin esto el arranque se cuelga descargando
   - `FLASK_PORT=8504`
   - `VL_PIPELINE_VERSION=v1`
   - `VL_RENDER_DPI=200`
3. Primer arranque: descarga el modelo VL (varios GB) → puede tardar minutos. Healthcheck
   con `start_period` de 10 min.
4. Verificar:
   ```bash
   CID=$(sudo docker ps -q -f name=gesfac-exp-paddleocrvl | head -1)
   sudo docker exec "$CID" curl -s http://localhost:8504/health   # -> vl_ready:true
   ```

## Endpoints
- `GET /health` → `{status, vl_ready, load_error}`
- `POST /process` (multipart `file=@factura.pdf`) → `{success, markdown, json, stats}`

## Benchmark (cuando esté healthy)
Mismo patrón que Fase 1 (ver `../../score_fase1.py` y `../../bench_runner.py`):
copiar las 21 facturas al contenedor, llamar a `/process`, guardar `markdown`+`json`.
Métrica de Fase 3 (distinta de Fase 1):
1. **Presencia de campos** en el markdown (reutilizar el scorer sobre el campo `markdown`).
2. **Extraíbilidad** (lo que de verdad importa): ¿el markdown está lo bastante ordenado
   para sacar campos sin ambigüedad? ¿separa emisor de receptor? ¿tablas bien formadas?
3. **¿Quita Gemini?** sí / no / parcial.

## Notas / riesgos
- La API exacta de `PaddleOCRVL` (atributos vs `save_to_*`, formato del JSON) puede
  necesitar ajuste tras el primer arranque real — igual que ajustamos los gotchas de
  Fase 1. El `app.py` lee los artefactos de `save_to_markdown`/`save_to_json` para ser
  robusto frente a nombres de salida.
- Vigilar VRAM total con `nvidia-smi` si `ia_ollama` tiene un modelo cargado a la vez.
