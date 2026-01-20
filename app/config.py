# Model options (speed vs accuracy trade-off):
# - "ArcFace" - Most accurate, slower (~512 dimensions)
# - "Facenet512" - Fast, very good accuracy (~512 dimensions) ✓ RECOMMENDED
# - "Facenet" - Faster, good accuracy (~128 dimensions)
MODEL_NAME = "Facenet512"

# Detector backend options (speed vs accuracy trade-off):
# - "retinaface" - Most accurate, slowest (~1-2s per detection)
# - "opencv" - Fast, good accuracy (~0.1-0.3s per detection) ✓ RECOMMENDED
# - "ssd" - Fast, decent accuracy (~0.2-0.4s per detection)
# - "mtcnn" - Balanced, ~0.5s per detection
DETECTOR_BACKEND = "opencv"

EMBEDDING_DIR = "models/embeddings"

# Similarity threshold for face matching (0.0 to 1.0)
# Higher = stricter matching, Lower = more lenient
# Recommended: 0.4-0.5 for Facenet512, 0.6-0.7 for ArcFace
THRESHOLD = 0.4
