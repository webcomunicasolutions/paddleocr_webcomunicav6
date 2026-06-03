# Fase 3 — PaddleOCR-VL 1.6 (VLM end-to-end) ⭐

Aquí irá el `Dockerfile` + config para servir **PaddlePaddle/PaddleOCR-VL** con vLLM en GPU.
Imagen→JSON estructurado de una pieza (puede sustituir OCR + Gemini). Apache 2.0, cabe en 12GB.
Puerto sugerido: 9103. Criterio éxito: ≥ (baseline + Gemini) y elimina la llamada a Gemini.
Pendiente: replicar el patrón de reserva de GPU en Swarm que usa `ia_ollama` (preguntado a servidor_gesfac vía bridge).

_(pendiente de rellenar al arrancar la Fase 3)_
