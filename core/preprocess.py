"""preprocess module

Utilities for image preprocessing used in the project.
- validate_image(path): checks if image can be opened.
- repair_and_resize(path, output_path, size=(224,224)): loads, converts to RGB, resizes, saves.
"""
import os
from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def validate_image(image_path: Path) -> bool:
    """Return True if image can be opened and is not a palette with transparency."""
    try:
        with Image.open(image_path) as img:
            img.verify()
            # Re-open to check mode after verify
            with Image.open(image_path) as img2:
                if img2.mode == "P" and "transparency" in img2.info:
                    return False
        return True
    except Exception:
        return False

def repair_and_resize(image_path: Path, output_path: Path, size: tuple[int, int] = (224, 224)) -> None:
    """Open image, convert to RGB, resize, and save as JPEG to output_path."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="JPEG", quality=95)

import numpy as np
def load_and_preprocess_image(image_path: Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load an image, resize it, and convert to a NumPy array for prediction.
    
    Returns a tensor of shape (1, height, width, 3).
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        img_array = np.array(img)
        # Normalize to [0, 1] as is common for many models
        img_array = img_array.astype("float32") / 255.0
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
