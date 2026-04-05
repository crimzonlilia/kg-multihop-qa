"""Model loading and caching for GLiNER2 and SentenceTransformer."""

import os
import logging
import torch
import sys
import io

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)

# Singleton caches
_EXTRACTOR_CACHE = None
_EMBEDDER_CACHE = None

MODEL_NAME = "fastino/gliner2-base-v1"
EMBEDDER_MODEL = os.getenv("KG_SCHEMA_EMBED_MODEL", os.getenv("KG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

# Optional imports with graceful fallback
try:
    from gliner2 import GLiNER2
except ImportError:
    GLiNER2 = None
    logger.warning("GLiNER2 not installed")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logger.warning("SentenceTransformer not installed - dynamic schema grouping disabled")


def get_extractor(model_name: str = MODEL_NAME):
    """Get or load GLiNER2 model (singleton pattern) with GPU optimizations."""
    global _EXTRACTOR_CACHE
    
    if _EXTRACTOR_CACHE is None:
        if GLiNER2 is None:
            raise ImportError("GLiNER2 not installed. Install with: pip install gliner2")
        
        logger.info(f"Loading GLiNER2 model: {model_name}")
        
        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"  Device: {device}")
        
        # Load model (suppress stdout to avoid emoji icon errors on Windows)
        try:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()  # Suppress print statements
            _EXTRACTOR_CACHE = GLiNER2.from_pretrained(model_name)
            sys.stdout = old_stdout
        except Exception as e:
            sys.stdout = old_stdout
            raise
        
        _EXTRACTOR_CACHE = _EXTRACTOR_CACHE.to(device)
        
        # Apply GPU optimizations if available
        if device == "cuda":
            try:
                _EXTRACTOR_CACHE.half()
                logger.info("✓ Model moved to GPU with fp16 optimization")
            except:
                logger.info("✓ Model moved to GPU (fp16 not available)")
        else:
            logger.info("✓ Model loaded on CPU (CUDA unavailable)")
    
    return _EXTRACTOR_CACHE


def get_embedder(model_name: str = EMBEDDER_MODEL):
    """Get or load SentenceTransformer model (singleton pattern)."""
    global _EMBEDDER_CACHE
    
    if _EMBEDDER_CACHE is None:
        if SentenceTransformer is None:
            raise ImportError("SentenceTransformer not installed. Install with: pip install sentence-transformers")
        
        logger.info(f"Loading SentenceTransformer: {model_name}")
        # Suppress stdout to avoid verbose output
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            _EMBEDDER_CACHE = SentenceTransformer(model_name)
        finally:
            sys.stdout = old_stdout
    
    return _EMBEDDER_CACHE


def clear_model_cache():
    """Force reload models on next call (debug only)."""
    global _EXTRACTOR_CACHE, _EMBEDDER_CACHE
    _EXTRACTOR_CACHE = None
    _EMBEDDER_CACHE = None
    logger.info("Model caches cleared")
