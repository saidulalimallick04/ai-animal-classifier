import os
import random
import logging
import warnings
from pathlib import Path
from PIL import Image, ImageFile

# Allow loading of truncated images (some corrupted JPEGs can still be opened partially)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Suppress the specific PIL warning about palette images with transparency
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
    category=UserWarning,
    module="PIL.Image",
)

def is_image_valid(image_path: Path) -> bool:
    """Return True if the image can be opened and does not trigger known problems.

    The function skips:
    * Corrupted files that raise an exception on open/verify.
    * Palette ("P") images that contain a transparency entry (the warning the user saw).
    """
    try:
        with Image.open(image_path) as img:
            # Verify that Pillow can read the image data
            img.verify()
            # Re-open to inspect mode/info (verify() closes the file)
        with Image.open(image_path) as img:
            if img.mode == "P" and "transparency" in img.info:
                logging.warning(f"Skipping palette image with transparency: {image_path}")
                return False
        return True
    except Exception as e:
        logging.warning(f"Corrupted or unreadable image skipped: {image_path} ({e})")
        return False

def repair_and_save(image_path: Path, output_path: Path, target_size: tuple | None = None) -> None:
    """Open a valid image, optionally resize it, and save a clean JPEG copy.

    Args:
        image_path: Source image file.
        output_path: Destination path for the repaired image.
        target_size: Desired (width, height). If None, original dimensions are kept.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB to ensure a consistent colour space for JPEG
            img = img.convert("RGB")
            if target_size:
                img = img.resize(target_size, Image.LANCZOS)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format="JPEG", quality=95)
            logging.info(f"Repaired and saved: {output_path}")
    except Exception as e:
        logging.error(f"Failed to repair image {image_path}: {e}")

def process_images(input_root: Path, train_root: Path, test_root: Path, target_size: tuple | None = None, test_percentage: float = 0.2, seed: int = 42) -> None:
    """Walk the input directory tree, filter out problematic images, and split them into train/test.

    Args:
        input_root: Root folder containing label subfolders with images.
        train_root: Destination root for training images.
        test_root: Destination root for validation/test images.
        target_size: Desired (width, height) for resizing; ``None`` keeps original size.
        test_percentage: Fraction of images per label to allocate to the test set (e.g., 0.2 for 20%).
        seed: Random seed for reproducible splits.
    """
    random.seed(seed)
    for dirpath, _, filenames in os.walk(input_root):
        # Determine the relative label directory (e.g., "1", "2", ...)
        rel_dir = Path(dirpath).relative_to(input_root)
        # Collect valid image paths for this directory
        valid_paths = []
        for filename in filenames:
            if not filename.lower().endswith(('.jpg', '.jpeg')):
                continue
            src_path = Path(dirpath) / filename
            if is_image_valid(src_path):
                valid_paths.append(src_path)
        if not valid_paths:
            continue
        # Shuffle and split
        random.shuffle(valid_paths)
        split_idx = int(len(valid_paths) * (1 - test_percentage))
        train_paths = valid_paths[:split_idx]
        test_paths = valid_paths[split_idx:]
        # Process training images
        for src_path in train_paths:
            out_dir = train_root / rel_dir
            out_path = out_dir / src_path.name
            repair_and_save(src_path, out_path, target_size=target_size)
        # Process test images
        for src_path in test_paths:
            out_dir = test_root / rel_dir
            out_path = out_dir / src_path.name
            repair_and_save(src_path, out_path, target_size=target_size)

if __name__ == "__main__":
    # Adjust these paths as needed for your project layout
    INPUT_ROOT = Path("./../dataset/images")   # e.g. e:/AI Projects/ai-animal-classifier/dataset/images
    TRAIN_ROOT = Path("./../dataset/repaired_image_train")
    TEST_ROOT = Path("./../dataset/repaired_image_test")
    # Example: resize all images to 224x224 for model compatibility; set to None to keep original size
    TARGET_SIZE = (224, 224)
    TEST_PERCENTAGE = 0.20  # 20% for validation

    logging.info(f"Starting preprocessing: {INPUT_ROOT} → {TRAIN_ROOT} (train), {TEST_ROOT} (test)")
    process_images(INPUT_ROOT, TRAIN_ROOT, TEST_ROOT, target_size=TARGET_SIZE, test_percentage=TEST_PERCENTAGE)
    logging.info("Image preprocessing completed.")
