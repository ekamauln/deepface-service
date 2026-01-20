from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import shutil
import uuid
import os
import logging

from app.face_service import generate_embedding, verify_faces, initialize_models
from app.config import EMBEDDING_DIR
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Verification Service")

os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs("tmp", exist_ok=True)

# In-memory embedding cache
_embedding_cache = {}
_cache_timestamp = 0

# Preload models on startup to eliminate cold starts
logger.info("Initializing face recognition service...")
initialize_models()


def load_embeddings():
    """Load all embeddings from disk"""
    embeddings = {}
    for file in os.listdir(EMBEDDING_DIR):
        if not file.endswith(".npy"):
            continue
        user_id = file.replace(".npy", "")
        try:
            embeddings[user_id] = list(
                __import__("numpy").load(f"{EMBEDDING_DIR}/{file}")
            )
        except (EOFError, ValueError) as e:
            print(f"Warning: Skipping corrupted embedding file {file}: {e}")
            continue
    return embeddings


def get_embeddings():
    """Get embeddings with in-memory caching"""
    global _embedding_cache, _cache_timestamp
    
    # Check if embeddings directory has been modified
    try:
        current_mtime = os.path.getmtime(EMBEDDING_DIR)
    except OSError:
        current_mtime = 0
    
    # Reload cache if directory was modified or cache is empty
    if current_mtime > _cache_timestamp or not _embedding_cache:
        logger.info("Reloading embeddings from disk...")
        _embedding_cache = load_embeddings()
        _cache_timestamp = current_mtime
        logger.info(f"Loaded {len(_embedding_cache)} embeddings into cache")
    
    return _embedding_cache


@app.post("/verify")
async def verify(image: UploadFile = File(...)):
    start_time = time.time()
    
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    filename = f"tmp/{uuid.uuid4()}.jpg"
    with open(filename, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        embeddings = get_embeddings()
        logger.info(f"Using {len(embeddings)} stored embeddings")
        user_id, score, matched = verify_faces(filename, embeddings)

        if matched:
            logger.info(
                f"✓ MATCH FOUND - User: {user_id}, Confidence: {score:.4f}")
        else:
            logger.info(
                f"✗ NO MATCH - Best score: {score:.4f} (threshold: 0.5)")

    finally:
        os.remove(filename)
        elapsed = time.time() - start_time
        logger.info(f"⏱️  Verify request completed in {elapsed:.2f}s")

    if not matched:
        return {"matched": False}

    return {"matched": True, "userId": user_id, "confidence": score}


@app.post("/register")
async def register_face(
    user_id: str = Form(...), image: UploadFile = File(...)
):
    start_time = time.time()
    
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    file_path = f"tmp/{user_id}.jpg"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        embedding = generate_embedding(file_path)
        os.makedirs(EMBEDDING_DIR, exist_ok=True)

        import numpy as np
        np.save(f"{EMBEDDING_DIR}/{user_id}.npy", np.array(embedding))

    finally:
        os.remove(file_path)
        elapsed = time.time() - start_time
        logger.info(f"⏱️  Register request completed in {elapsed:.2f}s")

    return {"status": "registered", "userId": user_id}
