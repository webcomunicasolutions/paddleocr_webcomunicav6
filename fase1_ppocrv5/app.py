#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddlePaddle CPU Document Preprocessor
Prepara documentos para OCR con deteccion de orientacion y correccion
"""

import os
import sys
import json
import subprocess
import logging
import time
import math
import tempfile
import threading  # v5.5: Thread-safety para ocr_instance
import uuid  # v5.5: UUID único para archivos temporales
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename  # v5.3: Para prevenir path traversal

# Configurar logging ANTES de cualquier otra cosa
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

logger.info("[DEBUG] Iniciando imports...")

# CONFIGURAR DIRECTORIOS PADDLE ANTES DE IMPORTAR
os.environ['PADDLE_HOME'] = '/home/n8n/.paddleocr'
os.environ['PADDLEX_HOME'] = '/home/n8n/.paddlex'
# Forzar directorio home para evitar problemas
os.environ['HOME'] = '/home/n8n'

logger.info("[DEBUG] Variables de entorno configuradas")

# v5.3: Verificar imports (ya importados arriba, solo logging de confirmación)
try:
    logger.info(f"[DEBUG] OpenCV version: {cv2.__version__}")
except Exception as e:
    logger.error(f"[DEBUG] Error verificando OpenCV: {e}")

try:
    logger.info(f"[DEBUG] NumPy version: {np.__version__}")
except Exception as e:
    logger.error(f"[DEBUG] Error verificando NumPy: {e}")

logger.info("[DEBUG] Imports basicos completados")

# DIAGNOSTICO: Importar PaddleOCR paso a paso
try:
    logger.info("[DEBUG] Importando paddle...")
    import paddle
    logger.info(f"[DEBUG] Paddle version: {paddle.__version__}")
    logger.info(f"[DEBUG] Paddle device: {paddle.device.get_device()}")
except Exception as e:
    logger.error(f"[DEBUG] Error importando paddle: {e}")

try:
    logger.info("[DEBUG] Importando paddleocr...")
    import paddleocr
    logger.info(f"[DEBUG] PaddleOCR version: {paddleocr.__version__}")
except Exception as e:
    logger.error(f"[DEBUG] Error importando paddleocr: {e}")

try:
    logger.info("[DEBUG] Importando DocImgOrientationClassification...")
    from paddleocr import DocImgOrientationClassification
    logger.info("[DEBUG] DocImgOrientationClassification importado OK")
except Exception as e:
    logger.error(f"[DEBUG] Error importando DocImgOrientationClassification: {e}")

app = Flask(__name__)
# v5.3: Límite de tamaño de archivo (50MB) para prevenir DoS
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
logger.info("[DEBUG] Flask app creada")

# Variables configurables desde ENV
OPENCV_CONFIG = {
    'HSV_LOWER_H': int(os.getenv('OPENCV_HSV_LOWER_H', '0')),
    'HSV_LOWER_S': int(os.getenv('OPENCV_HSV_LOWER_S', '0')),
    'HSV_LOWER_V': int(os.getenv('OPENCV_HSV_LOWER_V', '140')),
    'HSV_UPPER_H': int(os.getenv('OPENCV_HSV_UPPER_H', '180')),
    'HSV_UPPER_S': int(os.getenv('OPENCV_HSV_UPPER_S', '60')),
    'HSV_UPPER_V': int(os.getenv('OPENCV_HSV_UPPER_V', '255')),
    'MIN_AREA_PERCENT': float(os.getenv('OPENCV_MIN_AREA_PERCENT', '0.15')),
    'MAX_AREA_PERCENT': float(os.getenv('OPENCV_MAX_AREA_PERCENT', '0.95')),
    'EPSILON_FACTOR': float(os.getenv('OPENCV_EPSILON_FACTOR', '0.01')),
    'ERODE_ITERATIONS': int(os.getenv('OPENCV_ERODE_ITERATIONS', '1')),
    'DILATE_ITERATIONS': int(os.getenv('OPENCV_DILATE_ITERATIONS', '2')),
    'MIN_WIDTH': int(os.getenv('OPENCV_MIN_WIDTH', '300')),
    'MIN_HEIGHT': int(os.getenv('OPENCV_MIN_HEIGHT', '400')),
    'EROSION_PERCENT': float(os.getenv('OPENCV_EROSION_PERCENT', '0.100')),
    'INNER_SCALE_FACTOR': float(os.getenv('OPENCV_INNER_SCALE_FACTOR', '1.12'))
}

ROTATION_CONFIG = {
    'MIN_CONFIDENCE': float(os.getenv('ROTATION_MIN_CONFIDENCE', '0.7')),
    'MIN_SKEW_ANGLE': float(os.getenv('ROTATION_MIN_SKEW_ANGLE', '0.2'))
}

# Configuracion OCR desde variables de entorno
# v5.4: Valores unificados con init_ocr() para evitar inconsistencias
OCR_CONFIG = {
    'ocr_work_dpi': int(os.getenv('OCR_WORK_DPI', '144')),
    'ocr_out_dpi': int(os.getenv('OCR_OUT_DPI', '72')),
    'text_det_thresh': float(os.getenv('OCR_TEXT_DET_THRESH', '0.25')),
    'text_det_box_thresh': float(os.getenv('OCR_TEXT_DET_BOX_THRESH', '0.4')),
    'text_det_unclip_ratio': float(os.getenv('OCR_TEXT_DET_UNCLIP_RATIO', '2.0')),
    'text_rec_score_thresh': float(os.getenv('OCR_TEXT_REC_SCORE_THRESH', '0.2')),
    'text_det_limit_side_len': int(os.getenv('OCR_TEXT_DET_LIMIT_SIDE_LEN', '960')),
    'text_det_limit_type': os.getenv('OCR_TEXT_DET_LIMIT_TYPE', 'min'),
    'text_recognition_batch_size': int(os.getenv('OCR_TEXT_RECOGNITION_BATCH_SIZE', '6')),
    'textline_orientation_batch_size': int(os.getenv('OCR_TEXTLINE_ORIENTATION_BATCH_SIZE', '1'))
}

# Inicializar DocPreprocessor y OCR globalmente
doc_preprocessor = None
ocr_instance = None
ocr_initialized = False

# v5.5: Semáforo para serializar peticiones OCR y evitar std::exception
# PaddleOCR NO es thread-safe - ver README.md para alternativas de escalado
ocr_semaphore = threading.Semaphore(1)
ocr_work_dpi = OCR_CONFIG['ocr_work_dpi']
ocr_out_dpi = OCR_CONFIG['ocr_out_dpi']


def init_docpreprocessor():
#    """Verificar versiones de PaddlePaddle e inicializar PP-LCNet_x1_0_doc_ori"""
    """Verificar versiones de PaddlePaddle e inicializar text_image_orientation"""
    global doc_preprocessor

    try:
        # Verificar versiones instaladas
        import paddle
        logger.info(f"[INIT] PaddlePaddle version: {paddle.__version__}")

        import paddleocr
        logger.info(f"[INIT] PaddleOCR version: {paddleocr.__version__}")

        # Verificar si estamos en CPU o GPU
        logger.info(f"[INIT] Paddle device: {paddle.device.get_device()}")
        logger.info(f"[INIT] CUDA available: {paddle.device.cuda.device_count()}")

        logger.info("[INIT] Inicializando DocImgOrientationClassification...")
        from paddleocr import DocImgOrientationClassification
        # Intentar con configuracion especifica para CPU
        doc_preprocessor = DocImgOrientationClassification(
            model_name="PP-LCNet_x1_0_doc_ori",
            device=os.getenv("OCR_DEVICE", "gpu")
        )
        logger.info("[OK] DocImgOrientationClassification inicializado correctamente")

        return True

    except Exception as e:
        logger.error(f"[ERROR] Error inicializando DocImgOrientationClassification: {e}")
        import traceback
        logger.error(f"[ERROR TRACEBACK] {traceback.format_exc()}")
        doc_preprocessor = None
        return False


def init_ocr():
    """Inicializar PaddleOCR con configuracion optimizada desde ENV"""
    global ocr_instance, ocr_initialized

    if ocr_initialized:
        return True

    try:
        logger.info("[OCR INIT] ==========================================================================================")
        logger.info("[OCR INIT]                                Inicializando PaddleOCR                                    ")
        logger.info("[OCR INIT] ==========================================================================================")

        # Verificar versiones
        import paddleocr
        import paddle
        from paddleocr import PaddleOCR

        # Leer configuracion desde ENV
        ocr_config = {
            'ocr_version': os.getenv('OCR_VERSION', 'PP-OCRv3'),       # Se ignora cuando se especifican model names
            'lang': os.getenv('OCR_LANG', 'en'),                       # Se ignora cuando se especifican model names
            'text_detection_model_name': os.getenv('OCR_TEXT_DETECTION_MODEL_NAME', None),
            'text_recognition_model_name': os.getenv('OCR_TEXT_RECOGNITION_MODEL_NAME', None),
            'use_doc_orientation_classify': os.getenv('OCR_USE_DOC_ORIENTATION', 'false').lower() == 'true',
            'use_doc_unwarping': os.getenv('OCR_USE_DOC_UNWARPING', 'false').lower() == 'true',
            'use_textline_orientation': os.getenv('OCR_USE_TEXTLINE_ORIENTATION', 'false').lower() == 'true',
            'text_det_thresh': float(os.getenv('OCR_TEXT_DET_THRESH', '0.1')),
            'text_det_box_thresh': float(os.getenv('OCR_TEXT_DET_BOX_THRESH', '0.4')),
            'text_det_limit_side_len': int(os.getenv('OCR_TEXT_DET_LIMIT_SIDE_LEN', '960')),
            'text_det_limit_type': os.getenv('OCR_TEXT_DET_LIMIT_TYPE', 'min'),
            'text_recognition_batch_size': int(os.getenv('OCR_TEXT_RECOGNITION_BATCH_SIZE', '6')),
            'text_det_unclip_ratio': float(os.getenv('OCR_TEXT_DET_UNCLIP_RATIO', '1.5')),
        }

        logger.info(f"[OCR INIT] PaddleOCR version:     {paddleocr.__version__}")
        logger.info(f"[OCR INIT] PaddlePaddle version:  {paddle.__version__}")
        logger.info(f"[OCR INIT] Dispositivo:           {paddle.device.get_device()}")
        logger.info("[OCR INIT] Configuracion:")
        logger.info(f"[OCR INIT]   Modelo:              {ocr_config['ocr_version']}")
        logger.info(f"[OCR INIT]     Deteccion:         {ocr_config['text_detection_model_name']}")
        logger.info(f"[OCR INIT]     Reconocimiento:    {ocr_config['text_recognition_model_name']}")
        logger.info(f"[OCR INIT]   Idioma:              {ocr_config['lang']}")
        logger.info(f"[OCR INIT]   Parametros:")
        logger.info(f"[OCR INIT]     Deteccion          (text_det_thresh):              {ocr_config['text_det_thresh']}")
        logger.info(f"[OCR INIT]     Umbral cajas       (text_det_box_thresh):          {ocr_config['text_det_box_thresh']}")
        logger.info(f"[OCR INIT]     Limite lado        (text_det_limit_side_len):      {ocr_config['text_det_limit_side_len']}px ({ocr_config['text_det_limit_type']})")
        logger.info(f"[OCR INIT]     Tamaño batch       (text_recognition_batch_size):  {ocr_config['text_recognition_batch_size']}")
        logger.info(f"[OCR INIT]     Orientacion doc    (use_doc_orientation_classify): {'SI' if ocr_config['use_doc_orientation_classify'] else 'NO'}")
        logger.info(f"[OCR INIT]     Distorsion         (use_doc_unwarping):            {'SI' if ocr_config['use_doc_unwarping'] else 'NO'}")
        logger.info(f"[OCR INIT]     Orientacion lineas (use_textline_orientation):     {'SI' if ocr_config['use_textline_orientation'] else 'NO'}")
        logger.info("[OCR INIT]")
        logger.info("[OCR INIT] Cargando modelos...")
        ocr_instance = PaddleOCR(**ocr_config)

        ocr_initialized = True
        logger.info("[OCR INIT] ==========================================================================================")
        logger.info("[OCR INIT] PaddleOCR inicializado correctamente")
        logger.info("[OCR INIT] Modelos cargados en memoria")
        logger.info("[OCR INIT] ==========================================================================================")
        return True

    except Exception as e:
        logger.error(f"[OCR INIT ERROR] Error inicializando PaddleOCR: {e}")
        import traceback
        logger.error(f"[OCR INIT ERROR] {traceback.format_exc()}")
        ocr_instance = None
        ocr_initialized = False
        return False

# Forzar inicializacion al inicio
logger.info("[START] Iniciando PaddlePaddle CPU Document Preprocessor...")
init_docpreprocessor()
logger.info("[START] Iniciando PaddleOCR...")
init_ocr()


def find_inner_rectangle(contour, image_shape, config):
    """
    Encuentra el cuadrilátero inscrito dentro del contorno usando erosión morfológica
    para eliminar penínsulas, pero preservando la forma trapezoidal si existe.
    Retorna tanto el trapezoide erosionado como el expandido.
    """
    try:
        # ========================================
        # PASO 1: Crear máscara y aplicar erosión
        # ========================================
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)

        min_dimension = min(image_shape[0], image_shape[1])
        target_erosion_pixels = int(min_dimension * config['EROSION_PERCENT'])

        kernel_size = max(5, int(target_erosion_pixels / 3))
        if kernel_size % 2 == 0:
            kernel_size += 1

        iterations = 3
        actual_erosion = kernel_size * iterations
        actual_percent = (actual_erosion / min_dimension) * 100

        logger.info(f"[IMG] [OCV] [BORDER] Erosion: kernel {kernel_size}x{kernel_size}, {iterations} iter = {actual_erosion}px ({actual_percent:.1f}%)")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask_eroded = cv2.erode(mask, kernel, iterations=iterations)

        # ========================================
        # PASO 2: Encontrar contorno de la máscara erosionada
        # ========================================
        eroded_contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not eroded_contours:
            logger.warning("[IMG] [OCV] [BORDER] No se encontraron contornos después de la erosión")
            return None, None, None, None, None, None

        largest_eroded = max(eroded_contours, key=cv2.contourArea)

        # ========================================
        # PASO 3: Aproximar a 4 puntos (preservar trapezoide)
        # ========================================
        epsilon = config['EPSILON_FACTOR'] * cv2.arcLength(largest_eroded, True)
        approx = cv2.approxPolyDP(largest_eroded, epsilon, True)

        if len(approx) != 4:
            for eps_mult in [0.02, 0.03, 0.04, 0.05, 0.01, 0.06, 0.07]:
                epsilon = eps_mult * cv2.arcLength(largest_eroded, True)
                approx = cv2.approxPolyDP(largest_eroded, epsilon, True)
                if len(approx) == 4:
                    break

        # ========================================
        # PASO 4: Obtener puntos erosionados (azul)
        # ========================================
        if len(approx) == 4:
            eroded_pts = approx.reshape(4, 2).astype("float32")
        else:
            points = np.array(largest_eroded).reshape(-1, 2)
            if len(points) > 4:
                rect = cv2.minAreaRect(points.astype(np.float32))
                eroded_pts = cv2.boxPoints(rect).astype("float32")
            else:
                return None, None, None, None, None, None

        # Ordenar puntos erosionados
        s = eroded_pts.sum(axis=1)
        diff = np.diff(eroded_pts, axis=1).flatten()
        tl = eroded_pts[np.argmin(s)]
        br = eroded_pts[np.argmax(s)]
        tr = eroded_pts[np.argmin(diff)]
        bl = eroded_pts[np.argmax(diff)]
        eroded_pts = np.array([tl, tr, br, bl], dtype="float32")

        # ========================================
        # PASO 5: Expandir para crear puntos finales (verde)
        # ========================================
        expanded_pts = eroded_pts.copy()

        if 'INNER_SCALE_FACTOR' in config and config['INNER_SCALE_FACTOR'] != 1.0:
            scale_factor = config['INNER_SCALE_FACTOR']

            # Calcular cuánto expandir en píxeles
            # Basado en el perímetro promedio del trapecio
            perimeter = (np.linalg.norm(eroded_pts[1] - eroded_pts[0]) + 
                        np.linalg.norm(eroded_pts[2] - eroded_pts[1]) +
                        np.linalg.norm(eroded_pts[3] - eroded_pts[2]) +
                        np.linalg.norm(eroded_pts[0] - eroded_pts[3]))

            # Expansión uniforme: cantidad de píxeles a expandir
            expansion_pixels = (perimeter / 4) * (scale_factor - 1.0)

            # Expandir cada lado perpendicularmente
            expanded_pts = []
            for i in range(4):
                p1 = eroded_pts[i]
                p2 = eroded_pts[(i + 1) % 4]
                p_prev = eroded_pts[(i - 1) % 4]
                p_next = eroded_pts[(i + 2) % 4]

                # Vector del lado actual
                side_vec = p2 - p1
                side_len = np.linalg.norm(side_vec)
                if side_len > 0:
                    side_unit = side_vec / side_len
                else:
                    side_unit = np.array([1, 0])

                # Vector perpendicular hacia afuera (rotación 90° antihoraria)
                perp = np.array([-side_unit[1], side_unit[0]])

                # Vector del lado anterior
                prev_vec = p1 - p_prev
                prev_len = np.linalg.norm(prev_vec)
                if prev_len > 0:
                    prev_unit = prev_vec / prev_len
                else:
                    prev_unit = np.array([1, 0])

                # Vector perpendicular del lado anterior
                prev_perp = np.array([prev_unit[1], prev_unit[0]])

                # Promedio de las perpendiculares para la esquina
                corner_direction = (perp + prev_perp) / 2
                corner_dir_len = np.linalg.norm(corner_direction)
                if corner_dir_len > 0:
                    corner_direction = corner_direction / corner_dir_len

                # Expandir el punto
                if i == 0 or i == 2:
                    # Para puntos 0 y 2, invertir el signo de la componente X de corner_direction
                    corner_direction[0] = -corner_direction[0]

                new_pt = p1 - corner_direction * expansion_pixels
                expanded_pts.append(new_pt)

            expanded_pts = np.array(expanded_pts, dtype="float32")

            logger.info(f"[IMG] [OCV] [BORDER] Expansión paralela aplicada: {scale_factor:.2f} ({(scale_factor-1)*100:.0f}%)")
            logger.info(f"[IMG] [OCV] [BORDER] Píxeles de expansión: {expansion_pixels:.1f}px")

        # ========================================
        # PASO 6: Calcular métricas
        # ========================================
        width_top = np.linalg.norm(expanded_pts[1] - expanded_pts[0])
        width_bottom = np.linalg.norm(expanded_pts[2] - expanded_pts[3])
        height_left = np.linalg.norm(expanded_pts[3] - expanded_pts[0])
        height_right = np.linalg.norm(expanded_pts[2] - expanded_pts[1])

        width_avg = (width_top + width_bottom) / 2
        height_avg = (height_left + height_right) / 2
        aspect_ratio = width_avg / height_avg if height_avg > 0 else 1
        aspect_factor = np.power(aspect_ratio, 1/25) if aspect_ratio > 0 else 1

        # Retornar ambos conjuntos de puntos: erosionados y expandidos
        return expanded_pts, eroded_pts, width_avg, height_avg, aspect_ratio, aspect_factor

    except Exception as e:
        logger.error(f"[IMG] [OCV] [BORDER] Error en find_inner_rectangle: {e}")
        return None, None, None, None, None, None


def set_img_dpi(img_file, dpi):
    """
    Actualizar metadatos DPI de una imagen
    
    Args:
        img_file (str): Ruta al archivo
        dpi (int): Nuevo valor DPI para los metadatos
    """
    try:
        from PIL import Image
        
        # Abrir imagen
        image = Image.open(img_file)
        
        # Guardar con nuevos metadatos DPI (sobrescribe el archivo)
        image.save(img_file, dpi=(dpi, dpi))
        
        logger.info(f"[UPDATE_DPI] Metadatos DPI actualizados en {img_file} a {dpi} DPI")
        
    except Exception as e:
        logger.error(f"[UPDATE_DPI] Error actualizando DPI en {img_file}: {e}")
        raise


def det_borders(image_path, npy_file, config):
    """
    Detectar contorno del papel y guardar puntos con visualización de tres niveles:
    - Rojo: contorno original
    - Azul: erosionado (sin penínsulas)
    - Verde: expandido final
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error("FALLO: No se pudo leer la imagen")
            return False, None

        visualization = image.copy()
        original_area = image.shape[0] * image.shape[1]
        logger.info(f"[IMG] [OCV] [BORDER] Imagen original: {image.shape[1]}x{image.shape[0]} pixels")

        # Convertir a HSV y crear máscara
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        HSV_LOWER = np.array([config['HSV_LOWER_H'], config['HSV_LOWER_S'], config['HSV_LOWER_V']])
        HSV_UPPER = np.array([config['HSV_UPPER_H'], config['HSV_UPPER_S'], config['HSV_UPPER_V']])
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=config['ERODE_ITERATIONS'])
        mask = cv2.dilate(mask, kernel, iterations=config['DILATE_ITERATIONS'])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(visualization, contours, -1, (200, 200, 200), 2)
        logger.info(f"[IMG] [OCV] [BORDER] Total contornos encontrados: {len(contours)}")

        if contours:
            largest = max(contours, key=cv2.contourArea)
            detected_area = cv2.contourArea(largest)
            logger.info(f"[IMG] [OCV] [BORDER] Contorno detectado: {detected_area:.0f} pixels")

            # Dibujar contorno original en amarillo
            cv2.drawContours(visualization, [largest], -1, (0, 255, 255), 3)

            # ========================================
            # TRAPEZOIDE ROJO (original)
            # ========================================
            epsilon = config['EPSILON_FACTOR'] * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)

            if len(approx) == 4:
                outer_pts = approx.reshape(4, 2).astype("float32")
            else:
                rect = cv2.minAreaRect(largest)
                outer_pts = cv2.boxPoints(rect).astype("float32")

            # Ordenar y dibujar en rojo
            s = outer_pts.sum(axis=1)
            diff = np.diff(outer_pts, axis=1).flatten()
            tl = outer_pts[np.argmin(s)]
            br = outer_pts[np.argmax(s)]
            tr = outer_pts[np.argmin(diff)]
            bl = outer_pts[np.argmax(diff)]
            outer_pts = np.array([tl, tr, br, bl], dtype="float32")

            outer_pts_int = outer_pts.astype(int)
            cv2.polylines(visualization, [outer_pts_int], True, (0, 0, 255), 2)

            # ========================================
            # TRAPEZOIDES AZUL Y VERDE (erosionado y expandido)
            # ========================================
            expanded_pts, eroded_pts, width_side_in, height_side_in, aspect_ratio_in, aspect_factor_in = find_inner_rectangle(
                largest, image.shape, config
            )

            if expanded_pts is not None:
                # Dibujar trapezoide erosionado en AZUL
                eroded_pts_int = eroded_pts.astype(int)
                cv2.polylines(visualization, [eroded_pts_int], True, (255, 0, 0), 3)  # Azul

                # Dibujar trapezoide expandido en VERDE
                expanded_pts_int = expanded_pts.astype(int)
                cv2.polylines(visualization, [expanded_pts_int], True, (0, 255, 0), 4)  # Verde

                # Marcar vértices del verde (final)
                for i, pt in enumerate(expanded_pts_int):
                    cv2.circle(visualization, tuple(pt), 8, (0, 255, 0), -1)
                    cv2.circle(visualization, tuple(pt), 10, (255, 255, 255), 2)
                    cv2.putText(visualization, str(i), tuple(pt + [15, -10]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Calcular areas
                green_area = cv2.contourArea(expanded_pts)
                red_area = cv2.contourArea(outer_pts)

                # Usar el menor
                if red_area < green_area:
                    pts_final = outer_pts
                    logger.info(f"[IMG] [OCV] [BORDER] Usando trapezoide ROJO (mas pequeno): {red_area:.0f} < {green_area:.0f}")
                else:
                    pts_final = expanded_pts
                    logger.info(f"[IMG] [OCV] [BORDER] Usando trapezoide VERDE (mas pequeno): {green_area:.0f} <= {red_area:.0f}")

                detection_method = "eroded-expanded"

            else:
                # Fallback
                logger.warning("[IMG] [OCV] [BORDER] Fallback: usando contorno reducido")
                center = np.mean(outer_pts, axis=0)
                pts_final = []
                for pt in outer_pts:
                    new_pt = center + (pt - center) * 0.9
                    pts_final.append(new_pt)
                pts_final = np.array(pts_final, dtype=np.float32)

                pts_int = pts_final.astype(int)
                cv2.polylines(visualization, [pts_int], True, (0, 255, 0), 4)
                detection_method = "fallback"

            # ========================================
            # CALCULAR AREA FINAL Y VERIFICAR UMBRALES
            # ========================================
            final_area = cv2.contourArea(pts_final)
            area_percent = (final_area / original_area) * 100
            min_area_percent = config['MIN_AREA_PERCENT'] * 100
            max_area_percent = config['MAX_AREA_PERCENT'] * 100

            logger.info(f"[IMG] [OCV] [BORDER] Area final: {area_percent:.1f}% del area total")

            # Guardar visualización
            out_base = npy_file.replace('.npy', '')
            vis_filename = f"{out_base}.png"
            cv2.imwrite(vis_filename, visualization)
            logger.info(f"[IMG] [OCV] [BORDER] Imagen provisional: {vis_filename}")

            # Verificar si el area es demasiado pequena (ruido)
            if area_percent < min_area_percent:
                logger.error(f"[IMG] [OCV] [BORDER] FALLO: Area muy pequena {area_percent:.1f}% < {min_area_percent:.1f}%")
                return False, f"too_small|{area_percent:.1f}%|area_insuficiente"

            # Verificar si el area es demasiado grande (bypass)
            if area_percent >= max_area_percent:
                logger.info(f"[IMG] [OCV] [BORDER] BYPASS: Area final {area_percent:.1f}% >= {max_area_percent:.1f}% - usando imagen original")
                return False, f"bypass|{area_percent:.1f}%|no_processing"

            # Calcular ángulo
            dx = pts_final[1][0] - pts_final[0][0]
            dy = pts_final[1][1] - pts_final[0][1]
            paper_angle = math.degrees(math.atan2(dy, dx))
            if paper_angle < 0:
                paper_angle += 360

            # Añadir leyenda
            cv2.putText(visualization, "Metodo: " + detection_method, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(visualization, "Rojo: Original", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(visualization, "Azul: Erosionado", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(visualization, "Verde: Final", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(visualization, f"Area: {area_percent:.1f}%", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(visualization, f"Angulo: {paper_angle:.1f} deg", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Guardar puntos finales
            np.save(npy_file, pts_final)
            logger.info(f"[IMG] [OCV] [BORDER] Puntos guardados en {npy_file}")

            return True, f"{detection_method}|{area_percent:.1f}%|{paper_angle:.1f}deg"

        else:
            logger.error("FALLO: No se encontraron contornos")
            return False, None

    except Exception as e:
        logger.error(f"FALLO: {e}")
        return False, None


def fix_perspective(image_path, npy_file, perspective_file, config):
    """
    Corregir perspectiva aplicando factor de aspecto para compensar
    la expansión diferencial en dimensiones
    """
    try:
        image = cv2.imread(image_path)
        pts = np.load(npy_file)

        logger.info(f"[IMG] [OCV] [PERSPECTIVE] Aplicando correccion perspectiva")

        # Ordenar puntos
        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        src = np.array([tl, tr, br, bl], dtype="float32")

        # Calcular dimensiones base
        width_base = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        height_base = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

        # Aplicar compensación por aspecto si está configurado
        if 'ASPECT_COMPENSATION' in config and config['ASPECT_COMPENSATION']:
            aspect_ratio = width_base / height_base if height_base > 0 else 1
            aspect_factor = np.power(aspect_ratio, 1/25)

            # Ajustar dimensiones con el factor de aspecto
            # Nota: Aquí aplicamos la compensación inversa porque ya se aplicó en la expansión
            width = int(width_base / aspect_factor)
            height = int(height_base * aspect_factor)
            
            logger.info(f"[IMG] [OCV] [PERSPECTIVE] Compensación de aspecto aplicada: {aspect_factor:.3f}")
            logger.info(f"[IMG] [OCV] [PERSPECTIVE] Dimensiones: {width_base}x{height_base} -> {width}x{height}")
        else:
            width = width_base
            height = height_base
        
        # Aplicar límites mínimos
        width = max(width, config.get('MIN_WIDTH', 100))
        height = max(height, config.get('MIN_HEIGHT', 100))
        
        dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(src, dst)
        corrected = cv2.warpPerspective(image, M, (width, height), 
                                       flags=cv2.INTER_CUBIC, 
                                       borderMode=cv2.BORDER_REPLICATE)
        
        cv2.imwrite(perspective_file, corrected, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        return True, f"{width}x{height}"
        
    except Exception as e:
        logger.error(f"FALLO: {e}")
        return False, None


def fix_orientation(img_path, doc_preprocessor):
    """
    Detectar y corregir orientacion de imagen
    Returns: (success, orientation_degrees, confidence, rotated)
    """
    try:
        if not doc_preprocessor:
            logger.info("[IMG] [PADDLE] [ORIENTATION] Modelo no disponible")
            return False, 0, 0.0, False

        # v5.6: Semáforo para doc_preprocessor (PaddlePaddle no es thread-safe)
        with ocr_semaphore:
            output = doc_preprocessor.predict(img_path, batch_size=1)
        orientation = '0'
        confidence = 0.0

        for res in output:
            result_data = res.res if hasattr(res, 'res') else res
            if isinstance(result_data, dict):
                label_names = result_data.get('label_names', [])
                scores = result_data.get('scores', [])
                if label_names and scores:
                    orientation = label_names[0]
                    confidence = scores[0]

        # Rotar si es necesario
        rotated = False
        if orientation in ['90', '180', '270'] and confidence > ROTATION_CONFIG['MIN_CONFIDENCE']:
            img = cv2.imread(img_path)
            if orientation == '90':
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif orientation == '180':
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif orientation == '270':
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(img_path, img)
            rotated = True

        return True, int(orientation), confidence, rotated

    except Exception as e:
        logger.warning(f"[IMG] [PADDLE] [ORIENTATION] Error detectando orientacion: {e}")
        return False, 0, 0.0, False


def fix_deskew(img_path):
    """
    Detectar y corregir inclinacion de imagen usando ImageMagick
    Returns: (success, skew_angle, corrected)
    """
    try:
        # Detectar angulo de inclinacion
        # v5.3: Añadido timeout para evitar bloqueos
        result = subprocess.run(['convert', img_path, '-deskew', '45%', '-format', '%[deskew:angle]', 'info:'],
                              capture_output=True, text=True, timeout=60)

        if result.returncode != 0 or not result.stdout.strip():
            logger.warning("[IMG] [CONVERT] [DESKEW] Error detectando inclinacion")
            return False, 0.0, False

        skew_angle = result.stdout.strip()

        try:
            skew_float = float(skew_angle)
            skew_abs = abs(skew_float)

            corrected = False
            if skew_abs > ROTATION_CONFIG['MIN_SKEW_ANGLE']:
                deskewed_path = img_path.replace('.png', '_deskewed.png')
                # v5.3: Añadido timeout
                result = subprocess.run([
                    'convert', img_path,
                    '-background', 'white',
                    '-interpolate', 'bicubic',
                    '-deskew', '45%',
                    '-fuzz', '10%',
                    '+repage',
                    deskewed_path
                ], capture_output=True, text=True, timeout=60)

                if result.returncode == 0 and os.path.exists(deskewed_path):
                    subprocess.run(['mv', deskewed_path, img_path], timeout=30)
                    corrected = True
                else:
                    logger.warning("[IMG] [CONVERT] [DESKEW] Error aplicando correccion")
                    return False, skew_float, False

            return True, skew_float, corrected

        except ValueError:
            logger.warning(f"[IMG] [CONVERT] [DESKEW] No se pudo parsear angulo: {skew_angle}")
            return False, 0.0, False

    except Exception as e:
        logger.warning(f"[IMG] [CONVERT] [DESKEW] Error procesando inclinacion: {e}")
        return False, 0.0, False


def init_pdf_prep(n8nHomeDir, base_name, ext):
    """Preparacion inicial de PDF - desproteger y copiar"""
    try:
        filename = f"{base_name}{ext}"
        in_file = f"{n8nHomeDir}/in/{filename}"
        out_pdf = f"{n8nHomeDir}/ocr/{base_name}_2.0.preocr.pdf"

        logger.info(f"[PDF] Preparando PDF: {in_file}")

        # Leer configuracion del JSON
        json_file = f"{n8nHomeDir}/json/{filename}.json"
        empresaNif = ""

        if os.path.exists(json_file):
            try:
                result = subprocess.run(['jq', '-r', '.empresaNif // ""', json_file], capture_output=True, text=True)
                if result.returncode == 0:
                    empresaNif = result.stdout.strip()
                    # v5.3: No loguear contraseña por seguridad
                    logger.info(f"[JSON] empresaNif: {'*' * len(empresaNif) if empresaNif else 'N/A'}")
            except Exception as e:
                logger.warning(f"[JSON] Error leyendo JSON: {e}")

        # Verificar si esta protegido
        # v5.3: Añadido timeout
        result = subprocess.run(['pdfinfo', in_file], capture_output=True, text=True, timeout=30)

        if 'Incorrect password' in result.stderr and empresaNif:
            logger.info("[PDF] PDF protegido, desprotegiendo...")

            # Desproteger con empresaNif
            tmp_file = f"{in_file}_unlocked.pdf"
            # v5.3: Añadido timeout
            result = subprocess.run([
                'qpdf', '--password=' + empresaNif, '--decrypt',
                in_file, tmp_file
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0 and os.path.exists(tmp_file):
                # Mover archivo desprotegido
                subprocess.run(['mv', tmp_file, in_file], timeout=30)
                logger.info("[PDF] PDF desprotegido correctamente")
            else:
                logger.warning("[PDF] No se pudo desproteger PDF")

        # Copiar a directorio OCR
        subprocess.run(['cp', in_file, out_pdf], timeout=30)
        logger.info(f"[PDF] PDF copiado a {out_pdf}")

        return True

    except Exception as e:
        logger.error(f"[PDF ERROR] {e}")
        return False


def init_img_prep(n8nHomeDir, base_name, ext):
    """Preparacion inicial de imagen - perspectiva y crear PDF"""
    try:
        filename = f"{base_name}{ext}"
        in_file = f"{n8nHomeDir}/in/{filename}"
        out_pdf = f"{n8nHomeDir}/ocr/{base_name}_2.0.preocr.pdf"

        logger.info(f"[IMG] Preparando imagen: {in_file}")

        # Rutas para archivos intermedios
        npy_file = f"{n8nHomeDir}/ocr/{base_name}_1.1.borders.npy"
        ocv_img = f"{n8nHomeDir}/ocr/{base_name}_1.2.ocv.png"
        fallback_pdf_file = f"{n8nHomeDir}/ocr/{base_name}_1.4.ocv.pdf"

        # 1.1. Detectar bordes/contorno
        success, detect_result = det_borders(in_file, npy_file, OPENCV_CONFIG)
        if success:
            logger.info(f"[IMG] [OCV] [BORDER] Resultado OK - {detect_result}")
        else:
            logger.warning(f"[IMG] [OCV] [BORDER] Fallo en deteccion de bordes o estan fuera de los valores min/max")

        # 1.2. Corregir perspectiva (solo si 1.1. funciono)
        if os.path.exists(npy_file):
            success, perspective_result = fix_perspective(in_file, npy_file, ocv_img, OPENCV_CONFIG)
            if success:
                logger.info(f"[IMG] [OCV] [PERSPECTIVE] Resultado OK - {perspective_result} pixels")
            else:
                logger.warning("[IMG] [OCV] [PERSPECTIVE] Fallo en correccion de perspectiva")

        # 1.3. Crear PDF preocr
        if os.path.exists(ocv_img):
            # v5.3: Añadido timeout
            result = subprocess.run(['convert', ocv_img, '-quality', '85', '-sampling-factor', '2x2,1x1,1x1', '-interlace', 'JPEG', out_pdf], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                # Mostrar resumen
                final_size_result = subprocess.run(['identify', '-format', '%wx%h', ocv_img], capture_output=True, text=True, timeout=30)

                if final_size_result.returncode == 0:
                    final_size = final_size_result.stdout.strip()
                    logger.info(f"[IMG] [PDF] PDF creado con imagen procesada: {final_size} pixels")
            else:
                logger.error("[IMG] [PDF] Fallo al crear PDF con imagen procesada")

        # 1.3.1. Fallback: crear PDF con imagen original si no existe
        if not os.path.exists(out_pdf):
            # v5.3: Añadido timeout
            result = subprocess.run(['convert', in_file, '-quality', '85', '-sampling-factor', '2x2,1x1,1x1', '-interlace', 'JPEG', out_pdf], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                logger.info("[IMG] [PDF] PDF creado con imagen original")
            else:
                logger.error("[IMG] [PDF] Fallo al crear PDF con imagen original")
                return False

        return True

    except Exception as e:
        logger.error(f"[IMG ERROR] {e}")
        return False


def det_scanned(pdf_path, page_num=1):
    """
    Detectar si una pagina especifica es escaneada o vectorial

    Criterios (OR):
    - Es vectorial si tiene fuentes embebidas
    - Es vectorial si NO tiene ninguna imagen >80% del area de pagina

    Returns: True si es escaneada, False si es vectorial
    """
    try:
        import subprocess
        import fitz  # PyMuPDF

        # 1. Verificar fuentes embebidas
        # v5.3: Añadido timeout
        result = subprocess.run(
            ['pdffonts', '-f', str(page_num), '-l', str(page_num), pdf_path],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            logger.warning(f"[det_scanned] pdffonts fallo en pagina {page_num}")
            return True  # Asumir escaneada si hay error

        # Contar fuentes embebidas
        embedded_fonts = 0
        lines = result.stdout.splitlines()

        for line in lines[2:]:  # Saltar headers
            if line.strip():
                parts = line.split()
                if len(parts) >= 5 and parts[4] == 'yes':  # columna 'emb'
                    embedded_fonts += 1

        # Si hay fuentes embebidas, es vectorial
        if embedded_fonts > 0:
            logger.info(f"[DET_SCANNED] Detectada pagina VECTORIAL ({embedded_fonts} fuentes embebidas)")
            return False

        # 2. Si no hay fuentes embebidas, verificar imagenes con PyMuPDF
        try:
            pdf = fitz.open(pdf_path)

            # Verificar que la pagina existe
            if page_num > len(pdf):
                logger.warning(f"[det_scanned] Pagina {page_num} no existe")
                return False

            page = pdf[page_num - 1]  # PyMuPDF usa indice base 0

            # Obtener area de la pagina
            page_width = page.rect.width
            page_height = page.rect.height
            page_area = page_width * page_height
            threshold_percentage = 80.0

            logger.info(f"[DET_SCANNED] Pagina: {page_width:.0f}x{page_height:.0f} pts")

            # Obtener todas las imagenes de la pagina
            images = page.get_images(full=True)

            if not images:
                # Sin imagenes = probablemente vectorial puro
                logger.info(f"[DET_SCANNED] Detectada pagina VECTORIAL (sin imagenes, {embedded_fonts} fuentes embebidas)")
                pdf.close()
                return False

            # Verificar el tamano de cada imagen en la pagina
            has_large_image = False

            for img_index, img_info in enumerate(images):
                xref = img_info[0]

                # Obtener los rectangulos donde aparece esta imagen
                try:
                    img_rects = page.get_image_rects(xref)

                    for rect in img_rects:
                        # Calcular area de la imagen
                        img_area = rect.width * rect.height
                        percentage = (img_area / page_area) * 100

                        logger.debug(f"[DET_SCANNED] Imagen {img_index}: {rect.width:.0f}x{rect.height:.0f} pts = {percentage:.1f}% del area")

                        if percentage > threshold_percentage:
                            logger.info(f"[DET_SCANNED] Imagen grande detectada: {percentage:.1f}% del area de pagina")
                            has_large_image = True
                            break

                    if has_large_image:
                        break

                except Exception as e:
                    logger.debug(f"[DET_SCANNED] Error obteniendo rectangulos de imagen {img_index}: {e}")
                    continue

            pdf.close()

            # Determinar resultado
            if has_large_image:
                logger.info(f"[DET_SCANNED] Detectada pagina ESCANEADA (imagen >{threshold_percentage:.0f}% del area)")
                return True
            else:
                logger.info(f"[DET_SCANNED] Detectada pagina VECTORIAL (sin imagenes grandes, {embedded_fonts} fuentes embebidas)")
                return False

        except Exception as e:
            logger.warning(f"[DET_SCANNED] Error usando PyMuPDF: {e}")
            # Fallback: si no podemos verificar imagenes, asumir vectorial si no hay fuentes embebidas
            return False

    except Exception as e:
        logger.error(f"[DET_SCANNED] Error en pagina {page_num}: {e}")
        return True  # Asumir escaneada en caso de error


def extract_pdf_images(n8nHomeDir, base_name, in_pdf, out_png, target_dpi=144):
    """
    Extraer imagenes de PDF vectorial y crear PNG con solo imagenes posicionadas

    Args:
        n8nHomeDir: Directorio base de n8n
        base_name: Nombre base del archivo
        in_pdf: Path del PDF de entrada
        out_png: Path del PNG de salida
        target_dpi: DPI objetivo para el PNG de salida (default: 144)
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import io

        logger.info(f"[EXTRACT_IMAGES] Extrayendo imagenes de: {in_pdf}")
        logger.info(f"[EXTRACT_IMAGES] DPI objetivo: {target_dpi}")

        # Calcular factor de escala respecto a 72 DPI (base de PDF)
        scale_factor = target_dpi / 72.0

        # Abrir PDF con PyMuPDF
        pdf_document = fitz.open(in_pdf)

        # Procesar primera pagina (PDF individual)
        page = pdf_document[0]

        # Obtener dimensiones de la pagina en puntos
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height

        logger.info(f"[EXTRACT_IMAGES] Pagina original: {page_width:.1f}x{page_height:.1f} pts (72 DPI)")

        # Calcular dimensiones escaladas para el canvas
        canvas_width = int(page_width * scale_factor)
        canvas_height = int(page_height * scale_factor)

        logger.info(f"[EXTRACT_IMAGES] Canvas escalado: {canvas_width}x{canvas_height} px ({target_dpi} DPI)")

        # Obtener lista de imagenes en la pagina
        image_list = page.get_images(full=True)

        logger.info(f"[EXTRACT_IMAGES] Imagenes encontradas: {len(image_list)}")

        if not image_list:
            logger.warning(f"[EXTRACT_IMAGES] Sin imagenes en pagina, creando PNG vacio")
            # Crear PNG vacio con dimensiones escaladas
            empty_img = Image.new('RGB', (canvas_width, canvas_height), 'white')
            empty_img.save(out_png, dpi=(target_dpi, target_dpi))
            pdf_document.close()
            return

        # Crear imagen base con dimensiones escaladas (fondo blanco)
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

        # Contador de imagenes procesadas exitosamente
        images_processed = 0

        # Procesar cada imagen
        for img_index, img_info in enumerate(image_list):
            try:
                # img_info contiene: [xref, smask, width, height, bpc, colorspace, alt_colorspace, name, filter]
                xref = img_info[0]

                # Extraer la imagen
                try:
                    img_dict = pdf_document.extract_image(xref)
                    if not img_dict or "image" not in img_dict:
                        logger.warning(f"[EXTRACT_IMAGES] No se pudo extraer imagen {img_index+1} (xref={xref})")
                        continue

                    # Obtener datos de la imagen
                    img_data = img_dict["image"]
                    img_ext = img_dict["ext"]
                    orig_width = img_dict["width"]
                    orig_height = img_dict["height"]

                    # Crear PIL Image desde los bytes
                    pil_img = Image.open(io.BytesIO(img_data))

                    logger.debug(f"[EXTRACT_IMAGES] Imagen {img_index+1} extraida: {orig_width}x{orig_height} {img_ext}")

                except Exception as e:
                    logger.warning(f"[EXTRACT_IMAGES] Error extrayendo imagen {img_index+1}: {e}")
                    continue

                # Obtener las posiciones de esta imagen en la pagina usando get_image_rects
                try:
                    img_rects = page.get_image_rects(xref)

                    if not img_rects:
                        logger.warning(f"[EXTRACT_IMAGES] No se encontraron posiciones para imagen {img_index+1}")
                        continue

                    # Procesar cada instancia de la imagen (puede aparecer varias veces)
                    for inst_index, rect in enumerate(img_rects):
                        # rect es un fitz.Rect con coordenadas en puntos PDF (72 DPI)
                        x0 = rect.x0
                        y0 = rect.y0
                        rect_width = rect.width
                        rect_height = rect.height

                        # Escalar coordenadas y dimensiones segun el DPI objetivo
                        x_pos = int(x0 * scale_factor)
                        y_pos = int(y0 * scale_factor)
                        target_width = int(rect_width * scale_factor)
                        target_height = int(rect_height * scale_factor)

                        logger.debug(f"[EXTRACT_IMAGES] Imagen {img_index+1}.{inst_index+1}:")
                        logger.debug(f"[EXTRACT_IMAGES]   - Original en PDF: {int(rect_width)}x{int(rect_height)} en ({int(x0)}, {int(y0)}) pts")
                        logger.debug(f"[EXTRACT_IMAGES]   - Escalada a {target_dpi} DPI: {target_width}x{target_height} en ({x_pos}, {y_pos}) px")

                        # Validar dimensiones
                        if target_width <= 0 or target_height <= 0:
                            logger.warning(f"[EXTRACT_IMAGES] Dimensiones invalidas: {target_width}x{target_height}")
                            continue

                        # Redimensionar imagen al tamano escalado
                        try:
                            # Usar LANCZOS para mejor calidad al escalar
                            pil_img_resized = pil_img.resize((target_width, target_height), Image.LANCZOS)
                            logger.debug(f"[EXTRACT_IMAGES] Redimensionada de {orig_width}x{orig_height} a {target_width}x{target_height}")
                        except Exception as resize_err:
                            logger.warning(f"[EXTRACT_IMAGES] Error redimensionando: {resize_err}")
                            continue

                        # Verificar que la imagen cabe en el canvas escalado
                        if (x_pos + target_width > canvas_width) or (y_pos + target_height > canvas_height):
                            logger.warning(f"[EXTRACT_IMAGES] Imagen excede limites del canvas, ajustando")
                            # Ajustar si es necesario
                            if x_pos + target_width > canvas_width:
                                crop_width = canvas_width - x_pos
                                if crop_width > 0:
                                    pil_img_resized = pil_img_resized.crop((0, 0, crop_width, target_height))
                                    target_width = crop_width
                            if y_pos + target_height > canvas_height:
                                crop_height = canvas_height - y_pos
                                if crop_height > 0:
                                    pil_img_resized = pil_img_resized.crop((0, 0, target_width, crop_height))
                                    target_height = crop_height

                        # Pegar en canvas escalado
                        # Las coordenadas Y en PyMuPDF ya tienen origen arriba-izquierda (correcto para PIL)
                        canvas.paste(pil_img_resized, (x_pos, y_pos))
                        images_processed += 1

                        logger.debug(f"[EXTRACT_IMAGES] Imagen pegada en canvas en posicion ({x_pos}, {y_pos})")

                except Exception as e:
                    logger.warning(f"[EXTRACT_IMAGES] Error obteniendo posiciones de imagen {img_index+1}: {e}")
                    continue

            except Exception as e:
                logger.warning(f"[EXTRACT_IMAGES] Error procesando imagen {img_index+1}: {e}")
                continue

        # Informar resultado
        if images_processed == 0:
            logger.warning(f"[EXTRACT_IMAGES] No se pudo procesar ninguna imagen, creando PNG vacio")
            canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        else:
            logger.info(f"[EXTRACT_IMAGES] Imagenes procesadas exitosamente: {images_processed}")

        # Guardar PNG resultante con metadatos DPI
        canvas.save(out_png, dpi=(target_dpi, target_dpi))
        logger.info(f"[EXTRACT_IMAGES] PNG creado: {out_png} a {target_dpi} DPI")

        # Cerrar documento PDF
        pdf_document.close()

    except Exception as e:
        logger.error(f"[EXTRACT_IMAGES ERROR] Error extrayendo imagenes: {e}")
        # Crear PNG vacio como fallback con DPI por defecto
        from PIL import Image
        fallback_width = int(595 * (target_dpi / 72.0))
        fallback_height = int(842 * (target_dpi / 72.0))
        fallback_img = Image.new('RGB', (fallback_width, fallback_height), 'white')
        fallback_img.save(out_png, dpi=(target_dpi, target_dpi))


def create_spdf(n8nHomeDir, base_name, in_pdf, spdf, page_num):
    """
    Procesar una pagina individual:
      - orientacion (rotation)
      - analisis (scanned/vectorial)
      - OCR con PaddleOCR
      - crear PDF con OCR buscable
      - extraer texto existente de PDF vectorial
    
    Returns:
        list: Lista de lineas de texto (OCR + texto existente para vectoriales)
    """
    global ocr_instance, ocr_initialized, ocr_work_dpi, ocr_out_dpi
    logger.info(f"[CREATE_SPDF] Procesando: {in_pdf}")

    page_start_time = time.time()

    # Detectar tipo de pagina
    page_scanned = det_scanned(in_pdf)

    if page_scanned:

        # Extraer a imagen con ocr_work_dpi
        # v5.3: Añadido timeout
        subprocess.run(['pdftoppm', '-png', '-f', '1', '-l', '1', '-r', str(ocr_work_dpi), in_pdf, in_pdf.replace('.pdf', '')], check=True, timeout=120)

        # Detectar y corregir orientacion
        in_png = f"{n8nHomeDir}/ocr/{base_name}_2.2.p-{page_num}.png"
        out_png = f"{n8nHomeDir}/ocr/{base_name}_2.3.orientation.p-{page_num}.png"
        subprocess.run(['mv', in_pdf.replace('.pdf', '-1.png'), in_png], check=True, timeout=30)
        subprocess.run(['cp', in_png, out_png], check=True, timeout=30)
        logger.info(f"[ORIENTATION] Detectando orientacion pagina {page_num}...")
        success, degrees, conf, rotated = fix_orientation(out_png, doc_preprocessor)

        if success:
            action = " - CORREGIDO" if rotated else ""
            logger.info(f"[ORIENTATION] Pagina {page_num}: {degrees} grados (confianza: {conf:.3f}){action}")

        # Detectar y corregir inclinacion
        in_png = f"{n8nHomeDir}/ocr/{base_name}_2.3.orientation.p-{page_num}.png"
        out_png = f"{n8nHomeDir}/ocr/{base_name}_2.4.deskew.p-{page_num}.png"
        subprocess.run(['cp', in_png, out_png], check=True, timeout=30)
        logger.info(f"[DESKEW] Detectando inclinacion pagina {page_num}...")
        success, angle, corrected = fix_deskew(out_png)

        if success:
            action = " - CORREGIDO" if corrected else ""
            logger.info(f"[DESKEW] Pagina {page_num}: {angle:.2f} grados{action}")

    else:
        # Extraer imagenes a PNG temporal
        out_png = f"{n8nHomeDir}/ocr/{base_name}_2.4.deskew.p-{page_num}.png"
        extract_pdf_images(n8nHomeDir, base_name, in_pdf, out_png, ocr_work_dpi)

    # Ejecutar OCR sobre la imagen extraida y preparada de la pagina
    logger.info(f"[OCR] Ejecutando OCR en pagina {page_num}...")
    ocr_start = time.time()

    # Reintentos para OCR
    # v5.5: Semáforo serializa peticiones OCR para evitar std::exception
    # PaddleOCR no es thread-safe, solo permitimos 1 petición OCR a la vez
    page_ocr_result = None
    max_attempts = 5
    consecutive_std_errors = 0

    with ocr_semaphore:  # v5.5: Serializar acceso al modelo OCR
        for attempt in range(1, max_attempts + 1):
            try:
                page_ocr_result = ocr_instance.predict(out_png)
                ocr_time = time.time() - ocr_start
                if page_ocr_result and len(page_ocr_result) > 0:
                    texts = page_ocr_result[0].get('rec_texts', [])
                    scores = page_ocr_result[0].get('rec_scores', [])
                    avg_conf = sum(scores)/len(scores) if scores else 0
                    logger.info(f"[OCR] Pagina {page_num}: {len(texts)} bloques detectados")
                    logger.info(f"[OCR] Confianza promedio: {avg_conf:.3f}")
                    logger.info(f"[OCR] Tiempo OCR: {ocr_time:.2f}s")
                else:
                    logger.warning(f"[OCR] Pagina {page_num}: Sin texto detectado")
                    page_ocr_result = None
                break  # Exito, salir del bucle de reintentos

            except Exception as e:
                error_msg = str(e)
                logger.error(f"[OCR] Error en pagina {page_num} (intento {attempt}): {error_msg}")

                # Detectar si es std::exception
                if "std::exception" in error_msg:
                    consecutive_std_errors += 1

                    # Al segundo error consecutivo, reinicializar
                    if consecutive_std_errors >= 2:
                        logger.warning("[OCR] Detectados 2 errores std::exception consecutivos - reinicializando modelo OCR...")

                        # Reinicializar modelo (ya estamos dentro del semáforo)
                        ocr_instance = None
                        ocr_initialized = False

                        # Forzar recolección de basura
                        import gc
                        gc.collect()

                        # Reinicializar
                        if init_ocr():
                            logger.info("[OCR] Modelo OCR reinicializado exitosamente")
                            consecutive_std_errors = 0  # Resetear contador
                        else:
                            logger.error("[OCR] No se pudo reinicializar OCR")
                            raise Exception("OCR model corrupted and cannot reinitialize")

                # Esperar antes del siguiente intento
                if attempt < max_attempts:
                    logger.info(f"[OCR] Esperando 1 segundo antes del siguiente intento...")
                    time.sleep(1)
                else:
                    logger.error(f"[OCR] Error definitivo tras {max_attempts} intentos")
                    raise

    # Procesar resultado OCR
    if page_ocr_result and len(page_ocr_result) > 0:
        text_lines, confidences, coordinates = parse_paddleocr_result(page_ocr_result[0])
    else:
        text_lines, confidences, coordinates = [], [], []

    # Crear SPDF con texto buscable
    try:
        if page_scanned:
            # Base: imagen procesada
            compose_pdf_ocr(out_png, (text_lines, confidences, coordinates), spdf, True, ocr_out_dpi)
        else:
            # Base: PDF vectorial original
            compose_pdf_ocr(in_pdf, (text_lines, confidences, coordinates), spdf, False, ocr_out_dpi)
 
            # Para vectoriales, descartar text_lines del OCR y leer el PDF completo ya compuesto
            try:
                # v5.3: Añadido timeout
                result = subprocess.run(['pdftotext', '-raw', spdf, '-'], capture_output=True, text=True, timeout=60)
                if result.returncode == 0 and result.stdout.strip():
                    # Reemplazar completamente text_lines con el texto del PDF compuesto
                    text_lines = result.stdout.splitlines()
                    logger.info(f"[CREATE_SPDF] PDF vectorial: {len(text_lines)} lineas de texto extraidas del PDF final")
            except Exception as e:
                logger.warning(f"[CREATE_SPDF] Error extrayendo texto vectorial: {e}")
                # Si falla, mantener text_lines del OCR como fallback

        logger.info(f"[CREATE_SPDF] PDF con OCR guardado en: {spdf}")

    except Exception as e:
        logger.error(f"[CREATE_SPDF] Error creando PDF final: {e}")
        # Fallback: copiar PDF original
        subprocess.run(['cp', in_pdf, spdf], check=True, timeout=30)
        logger.info(f"[CREATE_SPDF] PDF original copiado como fallback: {spdf}")

    page_time = time.time() - page_start_time
    logger.info(f"[CREATE_SPDF] Pagina {page_num} completada en {page_time:.2f}s: {spdf}")

    return text_lines, confidences, coordinates


def proc_mpdf_ocr(n8nHomeDir, base_name, ext):
    """
    Procesar PDF multipagina:
      - extraer paginas
      - procesarlas individualmente con create_spdf
      - combinar paginas procesadas en final_pdf
      - acumular texto OCR de todas las paginas
    """
    global doc_preprocessor, ocr_instance

    try:
        in_pdf = f"{n8nHomeDir}/ocr/{base_name}_2.0.preocr.pdf"
        out_pdf = f"{n8nHomeDir}/ocr/{base_name}_3.0.ocr.pdf"
        final_pdf = f"{n8nHomeDir}/pdf/{base_name}{ext}.pdf"

        logger.info("[PROC_PDF_OCR] ==========================================================================================")
        logger.info(f"[PROC_PDF_OCR] Procesando: {in_pdf}")
        logger.info("[PROC_PDF_OCR] ==========================================================================================")

        # Verificar que exista el archivo
        if not os.path.exists(in_pdf):
            logger.error(f"[PROC_PDF_OCR] Archivo no encontrado: {in_pdf}")
            return False, "File not found", None

        # Verificar modelos inicializados
        # v5.3: Corregido bug crítico - funciones eran initialize_* (no existían)
        if not doc_preprocessor:
            logger.info("[PROC_PDF_OCR] Inicializando modelo de orientacion...")
            if not init_docpreprocessor():
                logger.warning("[PROC_PDF_OCR] Modelo de orientacion no disponible, continuando sin rotacion")

        if not ocr_instance:
            logger.info("[PROC_PDF_OCR] Inicializando PaddleOCR...")
            if not init_ocr():
                logger.error("[PROC_PDF_OCR] No se pudo inicializar PaddleOCR")
                return False, "OCR initialization failed", None

        # Obtener numero de paginas
        # v5.3: Añadido timeout
        result = subprocess.run(['pdfinfo', in_pdf], capture_output=True, text=True, timeout=30)
        pages = 1
        for line in result.stdout.splitlines():
            if "Pages:" in line:
                pages = int(line.split(":")[1].strip())
                break

        # Extraer paginas individuales en /ocr
        # v5.3: Añadido timeout
        subprocess.run(['pdfseparate', in_pdf, f'{n8nHomeDir}/ocr/{base_name}_2.1.p-%d.pdf'], check=True, timeout=120)
        logger.info(f"[PROC_PDF_OCR] Paginas ({pages}): {base_name}_2.1.p-1.pdf - {base_name}_2.1.p-{pages}.pdf")

        # Procesar cada pagina individualmente
        mpdf = []
        mpdf_text = []
        mpdf_confidences = []
        mpdf_coordinates = []
        total_start_time = time.time()

        for page in range(1, pages + 1):
            page_pdf = f"{n8nHomeDir}/ocr/{base_name}_2.1.p-{page}.pdf"
            spdf = f"{n8nHomeDir}/ocr/{base_name}_2.7.spdf-{page}.pdf"

            logger.info(f"[PROC_PDF_OCR] =================================  Iniciando pagina {page}/{pages}  ===================================")

            # Procesar pagina individual y obtener texto OCR + coordenadas
            spdf_text, spdf_conf, spdf_coords = create_spdf(n8nHomeDir, base_name, page_pdf, spdf, page)

            # Acumular texto, confidencias y coordenadas de esta pagina
            if spdf_text:
                mpdf_text.extend(spdf_text)
                mpdf_confidences.extend(spdf_conf if spdf_conf else [])
                mpdf_coordinates.extend(spdf_coords if spdf_coords else [])
                logger.info(f"[PROC_PDF_OCR] Pagina {page}: {len(spdf_text)} lineas de texto OCR")

            # Verificar que se creo correctamente
            if os.path.exists(spdf):
                mpdf.append(spdf)
            else:
                logger.error(f"[PROC_PDF_OCR] Error: No se creo {spdf}")
                return False, f"Failed to create page {page}", None

        # Combinar todas las paginas procesadas
        logger.info(f"[PROC_PDF_OCR] Combinando {len(mpdf)} paginas procesadas...")
        # v5.3: Añadido timeout
        subprocess.run(['pdfunite'] + mpdf + [out_pdf], check=True, timeout=120)
        out_size_kb = os.path.getsize(out_pdf) / 1024
        logger.info(f"[PROC_PDF_OCR] PDF combinado creado ({out_size_kb:.0f}kB): {out_pdf}")

        # Generar en ubicacion final
        subprocess.run(['cp', out_pdf, final_pdf], check=True, timeout=30)
        final_size_kb = os.path.getsize(final_pdf) / 1024
        logger.info(f"[PROC_PDF_OCR] PDF final creado ({final_size_kb:.0f}kB): {final_pdf}")

        # Calcular estadisticas consolidadas
        total_time = time.time() - total_start_time

        logger.info("[PROC_PDF_OCR] ==========================================================================================")
        logger.info(f"[PROC_PDF_OCR] Proceso completado exitosamente")
        logger.info(f"[PROC_PDF_OCR] Total paginas procesadas: {pages}")
        logger.info(f"[PROC_PDF_OCR] Total lineas de texto OCR: {len(mpdf_text)}")
        logger.info(f"[PROC_PDF_OCR] Tiempo total: {total_time:.2f}s")
        logger.info("[PROC_PDF_OCR] ==========================================================================================")

        return True, "Success", {
            'text_lines': mpdf_text,  # Texto OCR acumulado
            'confidences': mpdf_confidences,  # Confidencias acumuladas
            'coordinates': mpdf_coordinates,  # Coordenadas para layout
            'total_blocks': len(mpdf_text),
            'pages': pages,
            'processing_time': total_time
        }
        
    except Exception as e:
        logger.error(f"[PROC_PDF_OCR ERROR] Error critico: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False, str(e), None


def parse_paddleocr_result(ocr_result):
    """Procesar resultado OCR de PaddleOCR v3 (.predict())"""
    text_lines = []
    confidences = []
    coordinates_list = []

    if not ocr_result:
        return text_lines, confidences, coordinates_list

    try:
        logger.info("[OCR PROCESS] Procesando resultado OCR...")

        # PaddleOCR v3 devuelve diccionario con rec_texts, rec_scores, etc.
        if isinstance(ocr_result, dict):
            texts = ocr_result.get('rec_texts', [])
            scores = ocr_result.get('rec_scores', [])
            polys = ocr_result.get('rec_polys', [])

            for i, text in enumerate(texts):
                if text and text.strip():
                    text_lines.append(text.strip())
                    confidences.append(float(scores[i]) if i < len(scores) else 0.0)
                    if i < len(polys):
                        # Convertir numpy array a lista Python para JSON serializable
                        poly = polys[i]
                        if hasattr(poly, 'tolist'):
                            poly = poly.tolist()
                        coordinates_list.append(poly)
                    else:
                        coordinates_list.append([])

        # Si viene en lista (multipagina), procesar cada elemento
        elif isinstance(ocr_result, list):
            for page_result in ocr_result:
                if isinstance(page_result, dict):
                    texts = page_result.get('rec_texts', [])
                    scores = page_result.get('rec_scores', [])
                    polys = page_result.get('rec_polys', [])

                    for i, text in enumerate(texts):
                        if text and text.strip():
                            text_lines.append(text.strip())
                            confidences.append(float(scores[i]) if i < len(scores) else 0.0)
                            if i < len(polys):
                                # Convertir numpy array a lista Python para JSON serializable
                                poly = polys[i]
                                if hasattr(poly, 'tolist'):
                                    poly = poly.tolist()
                                coordinates_list.append(poly)
                            else:
                                coordinates_list.append([])

        logger.info(f"[OCR OK] Procesado: {len(text_lines)} bloques detectados")

    except Exception as e:
        logger.error(f"[OCR ERROR] Error procesando resultado OCR: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return text_lines, confidences, coordinates_list


def compose_pdf_ocr(base_source, ocr_data, out_spdf, is_scanned, out_dpi=72):
    """
    Crear PDF de una pagina con OCR superpuesto

    Args:
        base_source: Path a imagen PNG (escaneada) o PDF original (vectorial)
        ocr_data: Tupla (text_lines, confidences, coordinates) del OCR
        out_spdf: Path donde guardar el PDF resultante
        is_scanned: True para paginas escaneadas, False para vectoriales
    """
    try:
        import io
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from PIL import Image

        text_lines, confidences, coordinates = ocr_data

        logger.info(f"[COMPOSE_PDF] Creando PDF {'escaneado' if is_scanned else 'vectorial'} - Base: {base_source}")
        logger.info(f"[COMPOSE_PDF] OCR: {len(text_lines)} bloques de texto")

        # DETECCION UNIFICADA DE DPI
        # Para ambos flujos necesitamos saber el DPI del PNG procesado
        if is_scanned:
            # El PNG es directamente base_source
            png_path = base_source
        else:
            # Para vectorial, el PNG esta en una ruta relacionada
            png_path = base_source.replace('.pdf', '.png').replace('_2.1.p-', '_2.4.deskew.p-')

        # Detectar DPI del PNG
        try:
            image = Image.open(png_path)
            img_width_orig, img_height_orig = image.size  # Guardar dimensiones originales
            img_dpi = image.info.get('dpi', (144, 144))
            if isinstance(img_dpi, tuple):
                img_dpi = img_dpi[0]  # Usar DPI X
            logger.info(f"[COMPOSE_PDF] DPI detectado del PNG: {img_dpi}")
        except Exception as e:
            img_dpi = 144  # Valor por defecto
            logger.warning(f"[COMPOSE_PDF] No se pudo detectar DPI ({e}), asumiendo {img_dpi}")

        if is_scanned:
            # FLUJO ESCANEADO: Crear PDF base con ImageMagick + texto OCR superpuesto

            img_scaled = base_source.replace('_2.4.deskew.p-', '_2.5.scaled.p-').replace('.png', '.jpg')
            scale_factor = out_dpi / img_dpi
            new_width = int(img_width_orig * scale_factor)
            new_height = int(img_height_orig * scale_factor)
            logger.info(f"[COMPOSE_PDF] Escalando imagen de {img_width_orig}x{img_height_orig} @ {img_dpi}dpi a {new_width}x{new_height} @ {out_dpi}dpi")
    
            # Redimensionar la imagen fisicamente
            image = image.resize((new_width, new_height), Image.LANCZOS)
            # Convertir a RGB si es necesario (JPEG no soporta transparencia)
            if image.mode in ['RGBA', 'P']:
                image = image.convert('RGB')
            # Guardar como JPG con buena calidad
            image.save(img_scaled, format='JPEG', quality=90, optimize=True)

            # 1. Crear PDF base optimizado con ImageMagick (como _2.0.preocr.pdf)
            pdf_base = img_scaled.replace('_2.5.scaled.p-', '_2.6.p-').replace('.jpg', '.pdf')
            # v5.3: Añadido timeout
            result = subprocess.run(['convert', img_scaled, '-quality', '85', '-sampling-factor', '2x2,1x1,1x1', '-interlace', 'JPEG', pdf_base], capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"[COMPOSE_PDF] Error creando PDF base con ImageMagick: {result.stderr}")
                raise Exception("Failed to create base PDF with ImageMagick")

            logger.info(f"[COMPOSE_PDF] PDF base creado: {pdf_base}")

            # 2. Leer PDF base optimizado
            import PyPDF2
            # v5.4: Usar context manager para asegurar cierre del archivo
            with open(pdf_base, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                original_page = pdf_reader.pages[0]

                # Obtener dimensiones de la pagina
                media_box = original_page.mediabox
                page_width = float(media_box.width)
                page_height = float(media_box.height)

                logger.info(f"[COMPOSE_PDF] Dimensiones PDF: {page_width:.1f} x {page_height:.1f} pts")

                # 3. Crear capa de texto OCR invisible
                ocr_buffer = io.BytesIO()
                c = canvas.Canvas(ocr_buffer, pagesize=(page_width, page_height))

                # Calcular factor de escala: del PNG original (donde se hizo OCR) al PDF
                # Las coordenadas OCR vienen del PNG original a img_dpi
                scale_x = page_width / img_width_orig
                scale_y = page_height / img_height_orig

                logger.info(f"[COMPOSE_PDF] Factores de escala OCR->PDF: x={scale_x:.3f}, y={scale_y:.3f}")

                # Superponer texto OCR invisible
                for i, text in enumerate(text_lines):
                    if i < len(coordinates) and len(coordinates[i]) > 0:
                        confidence = confidences[i] if i < len(confidences) else 0.0

                        # Filtrar texto con baja confianza
                        if confidence < 0.3:
                            continue

                        try:
                            coords = coordinates[i]

                            # Calcular coordenadas del texto
                            x_coords = [point[0] for point in coords]
                            y_coords = [point[1] for point in coords]

                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)

                            # Convertir coordenadas de PNG original a PDF usando factores de escala
                            x_pdf = x_min * scale_x
                            y_pdf = page_height - (y_max * scale_y)
                            height_pdf = (y_max - y_min) * scale_y

                            # Calcular tamano de fuente
                            font_size = max(6, min(height_pdf * 0.8, 20))

                            # Dibujar texto invisible para busqueda
                            c.setFillColorRGB(1, 1, 1, alpha=0.01)  # Casi transparente
                            c.setFont("Helvetica", font_size)
                            c.drawString(x_pdf, y_pdf, text)

                        except Exception as e:
                            logger.debug(f"[COMPOSE_PDF] Error posicionando texto '{text}': {e}")
                            continue

                c.save()
                ocr_buffer.seek(0)

                # 4. Combinar PDF base con capa OCR
                from PyPDF2 import PdfWriter

                pdf_writer = PdfWriter()

                # Leer capa OCR
                ocr_pdf = PyPDF2.PdfReader(ocr_buffer)
                ocr_page = ocr_pdf.pages[0]

                # Superponer OCR sobre pagina original
                original_page.merge_page(ocr_page)
                pdf_writer.add_page(original_page)

                # Guardar resultado final
                buffer = io.BytesIO()
                pdf_writer.write(buffer)

        else:
            # FLUJO VECTORIAL: PDF original como base + texto OCR de imagenes

            import PyPDF2
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter

            # v5.4: Usar context manager para asegurar cierre del archivo
            with open(base_source, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                original_page = pdf_reader.pages[0]

                # Obtener dimensiones de la pagina original
                media_box = original_page.mediabox
                page_width = float(media_box.width)
                page_height = float(media_box.height)

                # Calcular factor de escala: del DPI del PNG a 72 DPI del PDF
                scale_factor = 72.0 / img_dpi
                logger.info(f"[COMPOSE_PDF] Factor de escala: {scale_factor:.3f} ({img_dpi} DPI -> 72 DPI)")

                # Crear PDF temporal con solo texto OCR
                ocr_buffer = io.BytesIO()
                c = canvas.Canvas(ocr_buffer, pagesize=(page_width, page_height))

                # Solo superponer texto OCR (de imagenes extraidas)
                for i, text in enumerate(text_lines):
                    if i < len(coordinates) and len(coordinates[i]) > 0:
                        confidence = confidences[i] if i < len(confidences) else 0.0

                        if confidence < 0.3:
                            continue

                        try:
                            coords = coordinates[i]

                            # Las coordenadas del OCR vienen del PNG a img_dpi
                            # Necesitamos escalarlas a 72 DPI para el PDF
                            x_coords = [point[0] for point in coords]
                            y_coords = [point[1] for point in coords]

                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)

                            # APLICAR FACTOR DE ESCALA a las coordenadas
                            x_pdf = x_min * scale_factor
                            y_pdf = page_height - (y_max * scale_factor)
                            height_pdf = (y_max - y_min) * scale_factor

                            font_size = max(6, min(height_pdf * 0.8, 20))

                            logger.debug(f"[COMPOSE_PDF] Texto '{text[:20]}...': orig({x_min:.0f},{y_min:.0f}) -> pdf({x_pdf:.0f},{y_pdf:.0f})")

                            # Dibujar texto invisible
                            c.setFillColorRGB(1, 1, 1, alpha=0.01)
                            c.setFont("Helvetica", font_size)
                            c.drawString(x_pdf, y_pdf, text)

                        except Exception as e:
                            logger.debug(f"[COMPOSE_PDF] Error posicionando texto vectorial '{text}': {e}")
                            continue

                # Asegurar que siempre hay una pagina aunque no haya texto
                if len(text_lines) == 0:
                    c.showPage()  # Crear pagina vacia

                c.save()
                ocr_buffer.seek(0)

                # Combinar PDF original con capa OCR
                from PyPDF2 import PdfWriter

                pdf_writer = PdfWriter()

                # Leer capa OCR
                ocr_pdf = PyPDF2.PdfReader(ocr_buffer)
                ocr_page = ocr_pdf.pages[0]

                # Superponer OCR sobre pagina original
                original_page.merge_page(ocr_page)
                pdf_writer.add_page(original_page)

                # Guardar resultado
                buffer = io.BytesIO()
                pdf_writer.write(buffer)

        # Guardar PDF final
        buffer.seek(0)
        with open(out_spdf, 'wb') as f:
            f.write(buffer.getvalue())

        logger.info(f"[COMPOSE_PDF] PDF con OCR guardado: {out_spdf}")

    except Exception as e:
        logger.error(f"[COMPOSE_PDF ERROR] Error creando PDF: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Fallback: copiar archivo base
        try:
            if is_scanned:
                # Crear PDF simple con la imagen
                from PIL import Image
                image = Image.open(base_source)
                image.save(out_spdf, "PDF", resolution=out_dpi)
                logger.info(f"[COMPOSE_PDF] Fallback: PDF simple creado desde imagen")
            else:
                # Copiar PDF original
                subprocess.run(['cp', base_source, out_spdf], check=True, timeout=30)
                logger.info(f"[COMPOSE_PDF] Fallback: PDF original copiado")
        except Exception as fallback_error:
            logger.error(f"[COMPOSE_PDF] Error en fallback: {fallback_error}")
            raise


# ============================================================================
# FORMATO LAYOUT SIMPLIFICADO (v5)
# Reconstruye estructura espacial del documento usando coordenadas OCR
# ============================================================================

import re as re_module

# Patrones de headers de tabla (multi-idioma: ES, EN, DE, FR, PT)
# v5.0 - Expandidos para mejor detección
TABLE_HEADER_PATTERNS = [
    # Código/Referencia
    r'(?i)(c[oó]digo|code|artikel|art[ií]culo|ref\.?|sku|item|producto)',
    # Descripción
    r'(?i)(descripci[oó]n|description|bezeichnung|concepto|produto|d[eé]signation|detalle)',
    # Cantidad
    r'(?i)(cantidad|quantity|qty|menge|cant\.?|uds\.?|qtd|unid\.?|pcs|units)',
    # Precio unitario
    r'(?i)(precio|price|preis|pvp|pre[cç]o|unit|p\.?\s*unit|tarifa|rate)',
    # Descuento
    r'(?i)(dto\.?|desc\.?|descuento|discount|rabatt|remise|dcto)',
    # Importe/Total línea
    r'(?i)(importe|amount|betrag|total|neto|valor|montant|subtotal|line\s*total)',
    # IVA/Impuestos
    r'(?i)(iva|vat|mwst|tva|tax|impuesto|%)',
    # Albarán/Pedido (común en facturas españolas)
    r'(?i)(alb\.?|ped\.?|albar[aá]n|pedido|order|n[uú]mero)',
    # Base imponible
    r'(?i)(base|base\s*imp\.?|taxable)',
]

# Patrón de precios - detecta formatos europeos y americanos
PRICE_PATTERN = re_module.compile(r'\d{1,3}(?:[.,]\d{3})*[,\.]\d{2}|\d+[,\.]\d{2}')

# Patrones de fin de tabla - detecta secciones de totales/pie
END_TABLE_PATTERNS = [
    r'(?i)(forma\s*de\s*pago|payment|zahlung|vencimiento|paiement|pagamento)',
    r'(?i)(base\s*imponible|subtotal|zwischensumme|sous-total|imponibile)',
    r'(?i)(iva\s*\d+|%\s*iva|vat\s*\d|mwst|tva)',
    r'(?i)(total\s*factura|invoice\s*total|gesamtbetrag|total\s*general|grand\s*total)',
    r'(?i)(iban|banco|bank|cuenta|account|ccc|swift|bic)',
    r'(?i)(garant[ií]a|warranty|garantie)',
    r'(?i)(observaciones|notes|remarks|bemerkungen)',
    r'(?i)(recargo|surcharge|zuschlag|equivalencia)',
    r'(?i)(domicilio|direcci[oó]n|address|adresse)',
]


# v5.2: Función para detectar filas de datos de tabla (Agente 2)
def is_potential_data_row(line):
    """
    Detecta si una línea parece ser fila de datos de tabla.
    Útil para tablas sin headers claros o con formato no estándar.
    """
    # Contiene números que parecen precios (formato X,XX o X.XX)
    has_prices = bool(re_module.search(r'\d+[,\.]\d{2}', line))
    # Tiene múltiples "columnas" separadas por espacios (mínimo 3 tokens)
    tokens = line.split()
    has_columns = len(tokens) >= 3
    # Contiene cantidades típicas de factura
    has_quantity = bool(re_module.search(r'\b[1-9]\d{0,2}\b', line))

    return has_prices and (has_columns or has_quantity)


def format_text_with_layout_simple(text_blocks, coordinates, page_width=200):
    """
    Reconstruye estructura espacial del documento (version simplificada v5).

    Args:
        text_blocks: Lista de textos detectados
        coordinates: Lista de poligonos/bboxes [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        page_width: Ancho de caracteres para la salida

    Returns:
        Texto formateado con estructura espacial y tablas con pipes
    """
    if not text_blocks or not coordinates:
        return '\n'.join(text_blocks) if text_blocks else ''

    # 1. Parsear coordenadas y crear bloques
    blocks = []
    for i, text in enumerate(text_blocks):
        if i < len(coordinates) and coordinates[i]:
            poly = coordinates[i]
            try:
                if isinstance(poly[0], (list, tuple)):
                    xs = [float(p[0]) for p in poly]
                    ys = [float(p[1]) for p in poly]
                else:
                    xs = [float(poly[j]) for j in range(0, min(len(poly), 8), 2)]
                    ys = [float(poly[j]) for j in range(1, min(len(poly), 8), 2)]

                blocks.append({
                    'text': text,
                    'x_min': min(xs),
                    'y_center': (min(ys) + max(ys)) / 2,
                    'height': max(ys) - min(ys)
                })
                continue
            except (IndexError, TypeError, ValueError):
                pass
        # Fallback
        blocks.append({'text': text, 'x_min': 0, 'y_center': i * 30, 'height': 20})

    if not blocks:
        return '\n'.join(text_blocks)

    # 2. Agrupar bloques en filas por Y similar
    # v5.2: Probado 1.2 y 0.9 (ambos REGRESIÓN) - 0.7 es óptimo
    avg_height = sum(b['height'] for b in blocks) / len(blocks)
    row_tolerance = avg_height * 0.7

    blocks_sorted = sorted(blocks, key=lambda b: b['y_center'])
    rows = []
    current_row = [blocks_sorted[0]]

    for block in blocks_sorted[1:]:
        row_y_avg = sum(b['y_center'] for b in current_row) / len(current_row)
        if abs(block['y_center'] - row_y_avg) < row_tolerance:
            current_row.append(block)
        else:
            rows.append(current_row)
            current_row = [block]
    rows.append(current_row)

    # 3. Detectar tabla
    # v5.2: Reducido de >=3 a >=2 para detectar tablas pequeñas (Agente 2)
    header_row_idx = -1
    data_rows = []

    if len(rows) >= 2:
        # Buscar fila de headers
        for row_idx, row in enumerate(rows):
            row_text = ' '.join(b['text'] for b in row).upper()
            matches = sum(1 for p in TABLE_HEADER_PATTERNS if re_module.search(p, row_text))
            if matches >= 3:
                header_row_idx = row_idx
                break

        # Buscar filas de datos (con precios, hasta fin de tabla)
        if header_row_idx >= 0:
            for row_idx, row in enumerate(rows):
                if row_idx <= header_row_idx:
                    continue
                row_text = ' '.join(b['text'] for b in row)

                # Verificar fin de tabla
                if any(re_module.search(p, row_text) for p in END_TABLE_PATTERNS):
                    break

                # v5.2: Usar is_potential_data_row() o PRICE_PATTERN (Agente 2)
                if PRICE_PATTERN.findall(row_text) or is_potential_data_row(row_text):
                    data_rows.append(row_idx)

    is_table = header_row_idx >= 0 and len(data_rows) >= 1

    # 4. Calcular columnas si hay tabla
    col_positions = []
    col_widths = []
    if is_table:
        header_row = rows[header_row_idx]
        col_positions = sorted([b['x_min'] for b in header_row])
        num_cols = len(col_positions)
        col_widths = [max(10, (page_width - num_cols - 1) // num_cols)] * num_cols

    # 5. Obtener dimensiones del documento
    all_x = [b['x_min'] for b in blocks]
    doc_width = max(all_x) - min(all_x) if all_x else 1
    x_offset = min(all_x)

    # 6. Generar salida
    output_lines = []

    for row_idx, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda b: b['x_min'])

        # Fila de tabla con pipes
        if is_table and (row_idx == header_row_idx or row_idx in data_rows):
            # Asignar bloques a columnas
            columns_content = [[] for _ in range(len(col_positions))]
            for block in row_sorted:
                # Columna mas cercana
                col_idx = min(range(len(col_positions)),
                             key=lambda i: abs(block['x_min'] - col_positions[i]))
                columns_content[col_idx].append(block['text'])

            # Formatear con pipes
            formatted = []
            for i, texts in enumerate(columns_content):
                text = ' '.join(texts)
                w = col_widths[i] if i < len(col_widths) else 12
                if len(text) > w:
                    text = text[:w-2] + '..'
                formatted.append(text.ljust(w))

            output_lines.append('|' + '|'.join(formatted) + '|')

            # Separador despues del header
            if row_idx == header_row_idx:
                output_lines.append('+' + '+'.join(['-' * w for w in col_widths]) + '+')

        else:
            # Posicionamiento espacial simple basado en X
            line = [' '] * page_width
            for block in row_sorted:
                rel_x = (block['x_min'] - x_offset) / doc_width if doc_width > 0 else 0
                rel_x = max(0, min(0.95, rel_x))
                char_start = max(0, min(int(rel_x * page_width), page_width - len(block['text']) - 1))

                for i, char in enumerate(block['text']):
                    pos = char_start + i
                    if pos < page_width and line[pos] == ' ':
                        line[pos] = char

            output_lines.append(''.join(line).rstrip())

    return '\n'.join(output_lines)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy' if (doc_preprocessor and ocr_initialized) else 'initializing',
        'preprocessor_ready': doc_preprocessor is not None,
        'ocr_ready': ocr_initialized,
        'opencv_config': OPENCV_CONFIG,
        'rotation_config': ROTATION_CONFIG
    })


@app.route('/ocr', methods=['POST'])
def ocr():
    """Endpoint OCR - procesa documento completo con orientacion y OCR"""
    global doc_preprocessor, ocr_instance, ocr_initialized, ROTATION_CONFIG
    start_time = time.time()

    try:
        # 1. VALIDACION Y SETUP
        filename_param = request.form.get('filename')
        if not filename_param:
            return jsonify({'error': 'filename required'}), 400

        # v5.4: Validar rutas para prevenir path traversal
        ALLOWED_BASE_DIRS = ['/home/n8n', '/tmp']

        # Extraer paths y configuracion
        if filename_param.startswith('/'):
            full_path = filename_param
            filename = Path(full_path).name
            n8nHomeDir = str(Path(full_path).parent.parent)
        else:
            filename = filename_param
            n8nHomeDir = request.form.get('n8nHomeDir', '/home/n8n')

        # v5.4: Validar que n8nHomeDir esté en directorios permitidos
        resolved_path = os.path.realpath(n8nHomeDir)
        is_allowed = any(resolved_path.startswith(allowed) for allowed in ALLOWED_BASE_DIRS)
        if not is_allowed:
            logger.warning(f"[OCR] Path traversal attempt blocked: {n8nHomeDir} -> {resolved_path}")
            return jsonify({'error': 'Invalid path', 'allowed_dirs': ALLOWED_BASE_DIRS}), 403

        # v5.4: Prevenir path traversal en filename
        if '..' in filename or filename.startswith('/'):
            logger.warning(f"[OCR] Invalid filename blocked: {filename}")
            return jsonify({'error': 'Invalid filename'}), 400

        base_name = Path(filename).stem
        ext = Path(filename).suffix.lower()

        logger.info("")
        logger.info("[OCR] ==========================================================================================")
        logger.info(f"[OCR] Procesando: {n8nHomeDir}/in/{filename}")
        logger.info("[OCR] ==========================================================================================")

        # Actualizar MIN_SKEW_ANGLE si se pasa como parametro
        min_angle_param = request.form.get('min_angle')
        if min_angle_param:
            try:
                ROTATION_CONFIG['MIN_SKEW_ANGLE'] = float(min_angle_param)
                logger.info(f"[OCR] MIN_SKEW_ANGLE actualizado a: {ROTATION_CONFIG['MIN_SKEW_ANGLE']}")
            except ValueError:
                logger.warning(f"[OCR] Valor invalido para min_angle: {min_angle_param}")
        else:
            # v5.4: Añadido valor por defecto para evitar TypeError si ENV no existe
            ROTATION_CONFIG['MIN_SKEW_ANGLE'] = float(os.getenv('ROTATION_MIN_SKEW_ANGLE', '0.2'))

        # VERIFICAR Y CARGAR MODELOS SI ES NECESARIO
        if not doc_preprocessor:
            logger.info("[OCR] Modelo de orientacion no cargado, inicializando...")
            if not init_docpreprocessor():
                logger.warning("[OCR] No se pudo cargar modelo de orientacion")

        if not ocr_instance:
            logger.info("[OCR] Modelo OCR no cargado, inicializando...")
            if not init_ocr():
                return jsonify({'error': 'OCR initialization failed'}), 503

        # Verificar que realmente funcionan los modelos
        try:
            # Test rapido para verificar que OCR responde
            test_result = ocr_instance.predict.__name__
        except Exception:  # v5.3: Corregido bare except
            logger.warning("[OCR] OCR instance no responde, reinicializando...")
            ocr_instance = None
            if not init_ocr():
                return jsonify({'error': 'OCR reinitialization failed'}), 503

        # Crear directorios necesarios
        os.makedirs(f"{n8nHomeDir}/ocr", exist_ok=True)
        os.makedirs(f"{n8nHomeDir}/pdf", exist_ok=True)

        # Verificar que existe archivo de entrada
        in_file = f"{n8nHomeDir}/in/{filename}"
        if not os.path.exists(in_file):
            return jsonify({'error': f'File not found: {in_file}'}), 404

        # PREPARAR ARCHIVO RECIBIDO
        if ext == '.pdf':
            # PREPARACION PDF
            if not init_pdf_prep(n8nHomeDir, base_name, ext):
                return jsonify({'error': 'PDF preparation failed'}), 500
        else:
            # PREPARACION IMAGEN (genera _2.0.preocr.pdf)
            if not init_img_prep(n8nHomeDir, base_name, ext):
                return jsonify({'error': 'Image preparation failed'}), 500

        # 3. PROCESAMIENTO OCR (orientacion + OCR integrado)
        logger.info("[OCR] Ejecutando procesamiento OCR completo...")
        success, message, ocr_data = proc_mpdf_ocr(n8nHomeDir, base_name, ext)

        if not success:
            logger.error(f"[OCR] Error en procesamiento: {message}")
            return jsonify({'error': message}), 500

        # 4. PREPARAR RESPUESTA
        end_time = time.time()
        duration = end_time - start_time

        # Extraer datos del OCR
        text_lines = ocr_data.get('text_lines', [])
        confidences = ocr_data.get('confidences', [])
        coordinates = ocr_data.get('coordinates', [])
        total_blocks = ocr_data.get('total_blocks', 0)
        pages = ocr_data.get('pages', 1)

        # Calcular estadisticas
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Texto plano
        extracted_text_plain = '\n'.join(text_lines)

        # Texto con layout espacial (si hay coordenadas)
        if coordinates and len(coordinates) > 0:
            try:
                extracted_text_layout = format_text_with_layout_simple(text_lines, coordinates)
            except Exception as e:
                logger.warning(f"[OCR] Error generando layout: {e}")
                extracted_text_layout = extracted_text_plain
        else:
            extracted_text_layout = extracted_text_plain

        logger.info("[OCR] ==========================================================================================")
        logger.info(f"[OCR STATS] Documento procesado correctamente - Paginas: {pages} - Tiempo {duration:.2f}s")
        logger.info("[OCR] ==========================================================================================")

        return jsonify({
            'success': True,
            'in_file': filename,
            'pdf_file': f"{base_name}.pdf",
            'extracted_text': extracted_text_plain,  # Compatibilidad
            'extracted_text_plain': extracted_text_plain,
            'extracted_text_layout': extracted_text_layout,
            'ocr_blocks': text_lines,
            'coordinates': coordinates,
            'stats': {
                'total_pages': pages,
                'total_blocks': total_blocks,
                'avg_confidence': round(avg_confidence, 3),
                'processing_time': round(duration, 2)
            }
        })

    except Exception as e:
        logger.error(f"[OCR ERROR] Error en endpoint OCR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ENDPOINT /process - API REST MINIMALISTA (v5)
# ============================================================================

# Estadisticas del servidor
server_stats = {
    'startup_time': time.time(),
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'total_processing_time': 0.0
}


@app.route('/process', methods=['POST'])
def process():
    """
    Endpoint REST minimalista para procesamiento OCR.
    Acepta: multipart/form-data con campo "file" (PDF/imagen)
    Parametro opcional: "format" = "normal" (default) | "layout"
    """
    global server_stats
    start_time = time.time()
    server_stats['total_requests'] += 1
    temp_file_path = None
    n8nHomeDir = '/home/n8n'

    try:
        # 1. VALIDAR ARCHIVO
        if 'file' not in request.files:
            server_stats['failed_requests'] += 1
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        if not file.filename:
            server_stats['failed_requests'] += 1
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        ext = Path(file.filename).suffix.lower()
        if ext not in ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            server_stats['failed_requests'] += 1
            return jsonify({'success': False, 'error': f'Unsupported format: {ext}'}), 400

        output_format = request.form.get('format', 'normal')

        # 2. GUARDAR ARCHIVO TEMPORAL
        os.makedirs(f"{n8nHomeDir}/in", exist_ok=True)
        # v5.3: Usar secure_filename para prevenir path traversal
        safe_name = secure_filename(file.filename)
        # v5.5: UUID único para evitar colisiones en peticiones concurrentes
        unique_id = uuid.uuid4().hex[:8]
        temp_filename = f"proc_{int(time.time())}_{unique_id}_{safe_name}"
        temp_file_path = f"{n8nHomeDir}/in/{temp_filename}"
        file.save(temp_file_path)

        # 3. LLAMAR A /ocr INTERNAMENTE
        with app.test_request_context('/ocr', method='POST', data={'filename': temp_file_path}):
            response = ocr()
            response_data, status_code = (response, 200) if not isinstance(response, tuple) else response
            ocr_result = response_data.get_json()

        # 4. PROCESAR RESULTADO
        if not ocr_result.get('success'):
            server_stats['failed_requests'] += 1
            return jsonify({
                'success': False,
                'error': ocr_result.get('error', 'OCR failed'),
                'timestamp': time.time()
            }), 500

        # Seleccionar texto segun formato
        if output_format == 'layout':
            text = ocr_result.get('extracted_text_layout') or ocr_result.get('extracted_text', '')
        else:
            text = ocr_result.get('extracted_text_plain') or ocr_result.get('extracted_text', '')

        processing_time = time.time() - start_time
        server_stats['successful_requests'] += 1
        server_stats['total_processing_time'] += processing_time

        return jsonify({
            'success': True,
            'format': output_format,
            'text': text,
            'stats': {
                'avg_confidence': ocr_result.get('stats', {}).get('avg_confidence', 0.0),
                'processing_time': round(processing_time, 3),
                'total_blocks': ocr_result.get('stats', {}).get('total_blocks', 0),
                'total_pages': ocr_result.get('stats', {}).get('total_pages', 1)
            },
            'timestamp': time.time()
        })

    except Exception as e:
        server_stats['failed_requests'] += 1
        logger.error(f"[PROCESS] Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500

    finally:
        # 5. LIMPIAR ARCHIVOS TEMPORALES
        if temp_file_path:
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                base = Path(temp_file_path).stem
                import glob
                for f in glob.glob(f"{n8nHomeDir}/ocr/{base}*") + glob.glob(f"{n8nHomeDir}/pdf/{base}*"):
                    try:
                        os.remove(f)
                    except Exception:  # v5.3: Corregido bare except
                        pass
            except Exception:  # v5.3: Corregido bare except
                pass


@app.route('/stats', methods=['GET'])
def stats():
    """Estadisticas del servidor"""
    return jsonify({
        'uptime_seconds': round(time.time() - server_stats['startup_time'], 1),
        'total_requests': server_stats['total_requests'],
        'successful_requests': server_stats['successful_requests'],
        'failed_requests': server_stats['failed_requests'],
        'avg_processing_time': round(
            server_stats['total_processing_time'] / server_stats['successful_requests'], 3
        ) if server_stats['successful_requests'] > 0 else 0
    })


# ============================================================================
# DASHBOARD WEB (v5)
# ============================================================================

@app.route('/')
def dashboard():
    """Dashboard web para probar OCR con documentacion de API"""
    return '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PaddleOCR WebComunica v5</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               background: #1a1a2e; color: #eee; min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00d4ff; margin-bottom: 10px; }
        .subtitle { color: #888; margin-bottom: 20px; }

        /* Tabs */
        .tabs { display: flex; gap: 5px; margin-bottom: 20px; border-bottom: 2px solid #16213e; }
        .tab { padding: 12px 24px; background: #16213e; border: none; color: #888; cursor: pointer;
               border-radius: 8px 8px 0 0; font-size: 15px; transition: all 0.2s; }
        .tab:hover { background: #1a2a4e; color: #ccc; }
        .tab.active { background: #0f3460; color: #00d4ff; font-weight: bold; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* OCR Tab */
        .upload-area { background: #16213e; border: 2px dashed #00d4ff; border-radius: 12px;
                       padding: 40px; text-align: center; margin-bottom: 20px; cursor: pointer; }
        .upload-area:hover { background: #1a2a4e; }
        .upload-area.dragover { background: #0f3460; border-color: #00ff88; }
        input[type="file"] { display: none; }
        .btn { background: #00d4ff; color: #1a1a2e; border: none; padding: 12px 30px;
               border-radius: 8px; font-size: 16px; cursor: pointer; font-weight: bold; }
        .btn:hover { background: #00a8cc; }
        .btn:disabled { background: #555; cursor: not-allowed; }
        .options { display: flex; gap: 20px; margin: 20px 0; justify-content: center; }
        .option { display: flex; align-items: center; gap: 8px; }
        .result { background: #16213e; border-radius: 12px; padding: 20px; margin-top: 20px; display: none; }
        .result h3 { color: #00d4ff; margin-bottom: 15px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat { background: #0f3460; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 24px; color: #00d4ff; font-weight: bold; }
        .stat-label { font-size: 12px; color: #888; }
        .text-output { background: #0a0a15; padding: 20px; border-radius: 8px;
                       white-space: pre-wrap; font-family: monospace; font-size: 13px;
                       max-height: 500px; overflow-y: auto; line-height: 1.4; }
        .loading { display: none; text-align: center; padding: 40px; }
        .spinner { border: 4px solid #333; border-top: 4px solid #00d4ff; border-radius: 50%;
                   width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #3d1515; border: 1px solid #ff4444; color: #ff6666; padding: 15px; border-radius: 8px; }

        /* API Docs Tab */
        .docs-section { background: #16213e; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .docs-section h2 { color: #00d4ff; margin-bottom: 15px; font-size: 20px; }
        .docs-section h3 { color: #00ff88; margin: 20px 0 10px 0; font-size: 16px; }
        .endpoint { background: #0f3460; border-radius: 8px; padding: 15px; margin: 10px 0; }
        .endpoint-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .method { padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 12px; }
        .method.get { background: #00ff88; color: #000; }
        .method.post { background: #ff9800; color: #000; }
        .endpoint-path { font-family: monospace; color: #fff; font-size: 15px; }
        .endpoint-desc { color: #aaa; font-size: 14px; margin-bottom: 10px; }
        .code-block { background: #0a0a15; padding: 15px; border-radius: 8px; font-family: monospace;
                      font-size: 13px; overflow-x: auto; margin: 10px 0; white-space: pre; }
        .param-table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 14px; }
        .param-table th { text-align: left; padding: 8px; background: #0a0a15; color: #00d4ff; }
        .param-table td { padding: 8px; border-bottom: 1px solid #333; }
        .param-name { font-family: monospace; color: #00ff88; }
        .param-type { color: #888; font-size: 12px; }
        .copy-btn { background: #333; border: none; color: #888; padding: 5px 10px; border-radius: 4px;
                    cursor: pointer; font-size: 12px; float: right; }
        .copy-btn:hover { background: #444; color: #fff; }
        .response-example { border-left: 3px solid #00d4ff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>PaddleOCR WebComunica v5</h1>
        <p class="subtitle">OCR minimalista con deteccion de tablas</p>

        <div class="tabs">
            <button class="tab active" onclick="showTab('ocr')">OCR</button>
            <button class="tab" onclick="showTab('docs')">API Docs</button>
        </div>

        <!-- OCR Tab -->
        <div id="ocr-tab" class="tab-content active">
            <div class="upload-area" id="dropZone" onclick="document.getElementById('fileInput').click()">
                <p style="font-size: 48px; margin-bottom: 15px;">📄</p>
                <p style="font-size: 18px; margin-bottom: 10px;">Arrastra un PDF o imagen aqui</p>
                <p style="color: #888;">o haz clic para seleccionar</p>
                <input type="file" id="fileInput" accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp">
            </div>

            <div class="options">
                <label class="option">
                    <input type="radio" name="format" value="layout" checked> Layout (tablas)
                </label>
                <label class="option">
                    <input type="radio" name="format" value="normal"> Normal (texto plano)
                </label>
            </div>

            <div style="text-align: center;">
                <button class="btn" id="processBtn" disabled>Procesar documento</button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Procesando documento...</p>
            </div>

            <div class="result" id="result">
                <h3>Resultado</h3>
                <div class="stats" id="stats"></div>
                <div class="text-output" id="textOutput"></div>
            </div>
        </div>

        <!-- API Docs Tab -->
        <div id="docs-tab" class="tab-content">
            <div class="docs-section">
                <h2>API REST - Documentacion</h2>
                <p style="color: #aaa; margin-bottom: 15px;">
                    Base URL: <code style="background: #0a0a15; padding: 3px 8px; border-radius: 4px;">http://localhost:8505</code>
                </p>
            </div>

            <!-- /process endpoint -->
            <div class="docs-section">
                <h2>Endpoint Principal: /process</h2>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method post">POST</span>
                        <span class="endpoint-path">/process</span>
                    </div>
                    <p class="endpoint-desc">Procesa un documento PDF o imagen y extrae el texto con OCR.</p>

                    <h3>Parametros (form-data)</h3>
                    <table class="param-table">
                        <tr><th>Parametro</th><th>Tipo</th><th>Requerido</th><th>Descripcion</th></tr>
                        <tr>
                            <td><span class="param-name">file</span></td>
                            <td><span class="param-type">File</span></td>
                            <td>Si</td>
                            <td>Archivo PDF, PNG, JPG, JPEG, TIFF o BMP</td>
                        </tr>
                        <tr>
                            <td><span class="param-name">format</span></td>
                            <td><span class="param-type">String</span></td>
                            <td>No</td>
                            <td><code>layout</code> (default) - Preserva estructura espacial y tablas<br>
                                <code>normal</code> - Texto plano sin formato</td>
                        </tr>
                    </table>

                    <h3>Ejemplo cURL</h3>
                    <div class="code-block">curl -X POST http://localhost:8505/process \\
  -F "file=@factura.pdf" \\
  -F "format=layout"</div>

                    <h3>Respuesta exitosa</h3>
                    <div class="code-block response-example">{
  "success": true,
  "format": "layout",
  "text": "... texto extraido ...",
  "stats": {
    "avg_confidence": 0.967,
    "processing_time": 12.5,
    "total_blocks": 157,
    "total_pages": 1
  },
  "timestamp": 1733664000.123
}</div>

                    <h3>Respuesta de error</h3>
                    <div class="code-block response-example">{
  "success": false,
  "error": "Descripcion del error"
}</div>
                </div>
            </div>

            <!-- /health endpoint -->
            <div class="docs-section">
                <h2>Endpoints de Estado</h2>

                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method get">GET</span>
                        <span class="endpoint-path">/health</span>
                    </div>
                    <p class="endpoint-desc">Verifica el estado del servidor y los modelos OCR.</p>

                    <h3>Ejemplo</h3>
                    <div class="code-block">curl http://localhost:8505/health</div>

                    <h3>Respuesta</h3>
                    <div class="code-block response-example">{
  "status": "healthy",
  "ocr_ready": true,
  "preprocessor_ready": true,
  "opencv_config": { ... },
  "rotation_config": { ... }
}</div>
                </div>

                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method get">GET</span>
                        <span class="endpoint-path">/stats</span>
                    </div>
                    <p class="endpoint-desc">Obtiene estadisticas de uso del servidor.</p>

                    <h3>Ejemplo</h3>
                    <div class="code-block">curl http://localhost:8505/stats</div>

                    <h3>Respuesta</h3>
                    <div class="code-block response-example">{
  "total_requests": 150,
  "successful_requests": 145,
  "failed_requests": 5,
  "uptime_seconds": 3600
}</div>
                </div>
            </div>

            <!-- /ocr endpoint -->
            <div class="docs-section">
                <h2>Endpoint n8n: /ocr</h2>
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method post">POST</span>
                        <span class="endpoint-path">/ocr</span>
                    </div>
                    <p class="endpoint-desc">Endpoint compatible con n8n. Procesa archivos desde el sistema de archivos del servidor.</p>

                    <h3>Parametros (JSON body)</h3>
                    <table class="param-table">
                        <tr><th>Parametro</th><th>Tipo</th><th>Descripcion</th></tr>
                        <tr>
                            <td><span class="param-name">filename</span></td>
                            <td><span class="param-type">String</span></td>
                            <td>Ruta al archivo en /home/n8n/in/</td>
                        </tr>
                        <tr>
                            <td><span class="param-name">n8nHomeDir</span></td>
                            <td><span class="param-type">String</span></td>
                            <td>Directorio base (default: /home/n8n)</td>
                        </tr>
                    </table>

                    <h3>Ejemplo</h3>
                    <div class="code-block">curl -X POST http://localhost:8505/ocr \\
  -H "Content-Type: application/json" \\
  -d '{"filename": "/home/n8n/in/documento.pdf"}'</div>
                </div>
            </div>

            <!-- Formatos -->
            <div class="docs-section">
                <h2>Formatos de Salida</h2>

                <h3>Layout (recomendado para facturas)</h3>
                <p style="color: #aaa; margin: 10px 0;">Preserva la estructura espacial del documento. Detecta y formatea tablas automaticamente.</p>
                <div class="code-block">|CODIGO    |DESCRIPCION      |CANTIDAD |PRECIO  |IMPORTE |
+----------+-----------------+---------+--------+--------+
|A001      |Producto ejemplo |2        |15,50   |31,00   |
|A002      |Otro producto    |1        |25,00   |25,00   |</div>

                <h3>Normal</h3>
                <p style="color: #aaa; margin: 10px 0;">Texto plano, cada linea separada por salto de linea.</p>
                <div class="code-block">FACTURA
Fecha: 08/12/2025
Cliente: Empresa SA
Total: 56,00 EUR</div>
            </div>

            <!-- Integracion -->
            <div class="docs-section">
                <h2>Ejemplos de Integracion</h2>

                <h3>Python</h3>
                <div class="code-block">import requests

url = "http://localhost:8505/process"
files = {"file": open("factura.pdf", "rb")}
data = {"format": "layout"}

response = requests.post(url, files=files, data=data)
result = response.json()

if result["success"]:
    print(f"Confianza: {result['stats']['avg_confidence']}")
    print(result["text"])</div>

                <h3>JavaScript (fetch)</h3>
                <div class="code-block">const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('format', 'layout');

const response = await fetch('http://localhost:8505/process', {
    method: 'POST',
    body: formData
});
const result = await response.json();
console.log(result.text);</div>

                <h3>n8n (HTTP Request Node)</h3>
                <div class="code-block">Method: POST
URL: http://paddleocr:8503/process
Body Type: Form-Data
  - file: {{ $binary.data }}
  - format: layout</div>
            </div>

            <!-- Limites -->
            <div class="docs-section">
                <h2>Limites y Recomendaciones</h2>
                <table class="param-table">
                    <tr><th>Parametro</th><th>Valor</th></tr>
                    <tr><td>Tamaño maximo archivo</td><td>50 MB</td></tr>
                    <tr><td>Formatos soportados</td><td>PDF, PNG, JPG, JPEG, TIFF, BMP</td></tr>
                    <tr><td>Tiempo tipico por pagina</td><td>5-20 segundos</td></tr>
                    <tr><td>Confianza promedio</td><td>92-97%</td></tr>
                </table>
            </div>
        </div>
    </div>

    <script>
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
        }

        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const processBtn = document.getElementById('processBtn');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        let selectedFile = null;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(e => {
            dropZone.addEventListener(e, ev => { ev.preventDefault(); ev.stopPropagation(); });
        });
        dropZone.addEventListener('dragover', () => dropZone.classList.add('dragover'));
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', e => {
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

        function handleFile(file) {
            selectedFile = file;
            dropZone.innerHTML = `<p style="font-size: 48px; margin-bottom: 15px;">✅</p>
                <p style="font-size: 18px;">${file.name}</p>
                <p style="color: #888;">${(file.size / 1024).toFixed(1)} KB</p>`;
            processBtn.disabled = false;
        }

        processBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            const format = document.querySelector('input[name="format"]:checked').value;
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('format', format);

            loading.style.display = 'block';
            result.style.display = 'none';
            processBtn.disabled = true;

            try {
                const response = await fetch('/process', { method: 'POST', body: formData });
                const data = await response.json();

                if (data.success) {
                    document.getElementById('stats').innerHTML = `
                        <div class="stat"><div class="stat-value">${data.stats.processing_time}s</div><div class="stat-label">Tiempo</div></div>
                        <div class="stat"><div class="stat-value">${(data.stats.avg_confidence * 100).toFixed(1)}%</div><div class="stat-label">Confianza</div></div>
                        <div class="stat"><div class="stat-value">${data.stats.total_blocks}</div><div class="stat-label">Bloques</div></div>
                        <div class="stat"><div class="stat-value">${data.stats.total_pages}</div><div class="stat-label">Paginas</div></div>`;
                    document.getElementById('textOutput').textContent = data.text;
                    result.style.display = 'block';
                } else {
                    document.getElementById('textOutput').innerHTML = `<div class="error">${data.error || 'Error desconocido'}</div>`;
                    result.style.display = 'block';
                }
            } catch (err) {
                document.getElementById('stats').innerHTML = '';
                document.getElementById('textOutput').innerHTML = `<div class="error">Error: ${err.message}</div>`;
                result.style.display = 'block';
            }
            loading.style.display = 'none';
            processBtn.disabled = false;
        });
    </script>
</body>
</html>'''


if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', '8503'))
    logger.info("[START] Iniciando PaddlePaddle CPU Document Preprocessor...")

    # Detectar si estamos en produccion
    if os.getenv('FLASK_ENV') == 'production':
        from waitress import serve
        logger.info("[READY] Iniciando servidor Waitress (produccion)")
        serve(app, host='0.0.0.0', port=port, threads=4)
    else:
        logger.info("[READY] Iniciando servidor Flask (desarrollo)")
        app.run(host='0.0.0.0', port=port, debug=False)

    logger.info(f"[READY] Servidor listo en puerto {port}")

