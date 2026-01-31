import numpy as np
import os
import tempfile
import logging
from PIL import Image
from deepface import DeepFace
from config import MODEL_NAME, DETECTOR_BACKEND, THRESHOLD

logger = logging.getLogger(__name__)


def initialize_models():
    """Preload DeepFace models on startup to avoid cold starts on first request"""
    logger.info("Preloading DeepFace models...")

    # Create a dummy image to trigger model loading
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    temp_path = tempfile.mktemp(suffix='.jpg')

    try:
        Image.fromarray(dummy_img).save(temp_path)
        DeepFace.represent(
            img_path=temp_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
        logger.info(
            f"✓ Model berhasil dibuat sebelumnya ({MODEL_NAME} + {DETECTOR_BACKEND})")
    except Exception as e:
        logger.warning(
            f"Pra-muatan model gagal (akan dimuat saat perminataan pertama): {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def cosine_similiarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def generate_embedding(image_path: str):
    result = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
    )
    first_result = result[0]
    if isinstance(first_result, dict):
        return first_result["embedding"]
    return first_result


def verify_faces(image_path: str, stored_embeddings: dict):
    incoming = generate_embedding(image_path)

    best_match = None
    best_score = 0

    for user_id, embedding in stored_embeddings.items():
        score = cosine_similiarity(incoming, embedding)
        if score > best_score:
            best_score = score
            best_match = user_id

    return best_match, best_score, best_score >= THRESHOLD
