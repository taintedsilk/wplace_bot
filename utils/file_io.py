# wplace_bot/utils/file_io.py
import json
import logging
import time
from pathlib import Path

from typing import List, Dict, Any

from wplace_bot.config import PROFILE_DATA_FILE, DATA_LOCK

def save_profile_data(profile_name: str, charges: int, max_charges: int):
    """Safely saves the charge data for a given profile to a JSON file."""
    with DATA_LOCK:
        data = load_profile_data() # Reuse loader to handle missing/corrupt file
        data[profile_name] = {
            "current_charges": charges,
            "max_charges": max_charges,
            "last_updated_timestamp": time.time()
        }
        try:
            with open(PROFILE_DATA_FILE, 'w') as f:
                json.dump(data, f, indent=4)
            logging.info(f"Saved data for profile '{profile_name}': {charges}/{max_charges} charges.")
        except IOError as e:
            logging.error(f"Failed to write profile data to {PROFILE_DATA_FILE}: {e}")

def load_profile_data() -> Dict[str, Any]:
    """Loads profile data from the JSON file."""
    if not Path(PROFILE_DATA_FILE).exists():
        return {}
    try:
        with open(PROFILE_DATA_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logging.warning(f"Could not read or parse '{PROFILE_DATA_FILE}'. Treating as empty.")
        return {}

def get_template_files(directory: str) -> List[Path]:
    """Finds all .png template files in a given directory."""
    template_dir = Path(directory)
    if not template_dir.is_dir():
        logging.warning(f"Template directory '{directory}' not found.")
        return []
    files = sorted(list(template_dir.glob("*.png")))
    logging.info(f"Found {len(files)} templates in '{directory}'.")
    return files