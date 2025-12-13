"""utils module for the core package

Provides generic helper functions used across the project.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("core.utils")

# ----------------------------------------------------------------------
# Configuration loader
# ----------------------------------------------------------------------
def load_json_config(config_path: Path) -> Dict[str, Any]:
    """Load a JSON configuration file.

    Args:
        config_path: Path to a JSON file.

    Returns:
        Parsed JSON as a Python dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Loaded config from %s", config_path)
    return config

# ----------------------------------------------------------------------
# Misc helpers
# ----------------------------------------------------------------------
def chunk_list(data: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split a list into equally‑sized chunks.

    Example:
        >>> chunk_list([1,2,3,4,5], 2)
        [[1, 2], [3, 4], [5]]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
