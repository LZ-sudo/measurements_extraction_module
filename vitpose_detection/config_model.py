"""
ViTPose Model Configuration

Shared model configuration for ViTPose detection modules.
Handles model definitions, caching, and downloading from Hugging Face Hub.
"""

import os
from pathlib import Path
from typing import Tuple

from huggingface_hub import hf_hub_download


# ============================================================================
# Model Configuration
# ============================================================================

# Hugging Face repository for easyViTPose models
HF_REPO_ID = "JunkyByte/easy_ViTPose"

# Model configurations: (repo_path, local_filename, description, approximate_size_mb)
# repo_path is the path within the Hugging Face repository
VITPOSE_MODELS = {
    's': ('torch/wholebody/vitpose-s-wholebody.pth', 'vitpose-s-wholebody.pth', 'ViTPose Small', 90),
    'b': ('torch/wholebody/vitpose-b-wholebody.pth', 'vitpose-b-wholebody.pth', 'ViTPose Base', 90),
    'l': ('torch/wholebody/vitpose-l-wholebody.pth', 'vitpose-l-wholebody.pth', 'ViTPose Large', 310),
    'h': ('torch/wholebody/vitpose-h-wholebody.pth', 'vitpose-h-wholebody.pth', 'ViTPose Huge', 350),
}

# YOLO model configurations: (repo_path, local_filename, description, approximate_size_mb)
YOLO_MODELS = {
    'yolov8s': ('yolov8/yolov8s.pt', 'yolov8s.pt', 'YOLOv8 Small', 22),
    'yolov8m': ('yolov8/yolov8m.pt', 'yolov8m.pt', 'YOLOv8 Medium', 52),
}

# Default models for maximum accuracy
DEFAULT_VITPOSE_MODEL = 'h'
DEFAULT_YOLO_MODEL = 'yolov8s'


def get_cache_dir() -> Path:
    """Get the cache directory for ViTPose models (~/.cache/vitpose/).

    Uses the user's home cache directory so model weights persist across
    setup.bat runs and are shared if multiple projects use easyViTPose.
    Can be overridden by setting the VITPOSE_CACHE environment variable.
    """
    cache_dir = os.getenv(
        'VITPOSE_CACHE',
        os.path.join(os.path.expanduser('~'), '.cache', 'vitpose')
    )
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def download_model(repo_path: str, local_filename: str, repo_id: str = HF_REPO_ID) -> str:
    """Download a model file from Hugging Face Hub if not cached.

    Args:
        repo_path: Path to file within the Hugging Face repository
        local_filename: Name to save the file as locally
        repo_id: Hugging Face repository ID

    Returns:
        Path to the downloaded/cached model file
    """
    cache_dir = get_cache_dir()
    cached_path = cache_dir / local_filename

    if cached_path.exists():
        return str(cached_path)

    print(f"Downloading {local_filename} from Hugging Face...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=repo_path,
        cache_dir=str(cache_dir),
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False
    )

    # Move from subdirectory to cache root if needed
    downloaded_path = Path(downloaded_path)
    if downloaded_path.name != local_filename or downloaded_path.parent != cache_dir:
        target_path = cache_dir / local_filename
        if downloaded_path.exists() and not target_path.exists():
            downloaded_path.rename(target_path)
        return str(target_path)

    return str(downloaded_path)


def get_model_paths(
    vitpose_model: str = DEFAULT_VITPOSE_MODEL,
    yolo_model: str = DEFAULT_YOLO_MODEL
) -> Tuple[str, str]:
    """
    Get paths to ViTPose and YOLO model files, downloading if necessary.

    Args:
        vitpose_model: ViTPose model size key ('s', 'b', 'l', or 'h').
        yolo_model: YOLO model key ('yolov8s' or 'yolov8m').

    Returns:
        Tuple of (vitpose_model_path, yolo_model_path) as strings.
    """
    if vitpose_model not in VITPOSE_MODELS:
        raise ValueError(f"Invalid ViTPose model '{vitpose_model}'. Available: {list(VITPOSE_MODELS.keys())}")
    if yolo_model not in YOLO_MODELS:
        raise ValueError(f"Invalid YOLO model '{yolo_model}'. Available: {list(YOLO_MODELS.keys())}")

    vitpose_repo_path, vitpose_local_filename = VITPOSE_MODELS[vitpose_model][:2]
    yolo_repo_path, yolo_local_filename = YOLO_MODELS[yolo_model][:2]

    vitpose_path = download_model(vitpose_repo_path, vitpose_local_filename)
    yolo_path = download_model(yolo_repo_path, yolo_local_filename)

    return vitpose_path, yolo_path
