# run_manager.py
import logging
import subprocess
import time
import threading
import random
from pathlib import Path
from typing import List, Dict, Any, Set

from wplace_bot.config import (
    PROFILE_DATA_FILE, CHARGE_GAIN_RATE_SECONDS, PROFILES_TO_USE,
    PROFILES_FOLDER, PROFILE_TIMEOUT_SECONDS, RUN_INTERVAL_SECONDS,
    RANDOM_TILE_CHECK_INTERVAL_SECONDS, TEMPLATE_DIRECTORY, IDLE_CHECK_INTERVAL_SECONDS
)
from wplace_bot.utils.file_io import load_profile_data, get_template_files
from wplace_bot.analysis.canvas_analyzer import CanvasAnalyzer

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MANAGER] [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler("profile_manager.log", mode='w'), logging.StreamHandler()]
)

# --- Shared State for Manager and Monitoring Thread ---
templates_to_fix: Set[Path] = set()
all_template_paths: List[Path] = []
state_lock = threading.Lock()

def check_template(template_path: Path) -> bool:
    """Analyzes a single template and returns True if it needs fixing."""
    try:
        analyzer = CanvasAnalyzer()
        analyzer.analyze_templates([template_path])
        return analyzer.needs_fixing()
    except Exception as e:
        logging.error(f"[Monitor] Error analyzing template {template_path.name}: {e}")
        return False

def canvas_monitoring_worker():
    """
    A worker thread that periodically checks the canvas state.
    On start, it checks all templates. Then, it checks a random template periodically.
    """
    global templates_to_fix, all_template_paths
    logging.info("[Monitor] Starting initial canvas scan of all templates...")
    
    initial_fix_list = [path for path in all_template_paths if check_template(path)]
    
    with state_lock:
        templates_to_fix.update(initial_fix_list)
        
    logging.info(f"[Monitor] Initial scan complete. Found {len(templates_to_fix)} templates needing fixes.")

    while True:
        try:
            time.sleep(RANDOM_TILE_CHECK_INTERVAL_SECONDS)
            if not all_template_paths:
                continue

            template_to_check = random.choice(all_template_paths)
            logging.info(f"[Monitor] Performing random check on: {template_to_check.name}")
            
            needs_fixing = check_template(template_to_check)
            
            with state_lock:
                if needs_fixing and template_to_check not in templates_to_fix:
                    logging.info(f"[Monitor] Found new template to fix: {template_to_check.name}")
                    templates_to_fix.add(template_to_check)
                elif not needs_fixing and template_to_check in templates_to_fix:
                    logging.info(f"[Monitor] Template is now fixed, removing from queue: {template_to_check.name}")
                    templates_to_fix.remove(template_to_check)

        except Exception as e:
            logging.error(f"[Monitor] An error occurred in the monitoring loop: {e}", exc_info=True)

def calculate_profile_statuses(all_data: Dict[str, Any], profiles_to_run: List[str]) -> List[Dict[str, Any]]:
    """Calculates the current estimated charges and identifies profiles at full charge."""
    statuses = []
    current_time = time.time()
    
    for name in profiles_to_run:
        data = all_data.get(name)
        if not data:
            statuses.append({"name": name, "estimated_charges": 0, "max_charges": 100, "is_full": False})
            continue

        seconds_since_update = current_time - data.get("last_updated_timestamp", current_time)
        charges_gained = seconds_since_update / CHARGE_GAIN_RATE_SECONDS
        estimated_charges = min(data["max_charges"], data["current_charges"] + charges_gained)
        is_full = estimated_charges >= data["max_charges"]
            
        statuses.append({
            "name": name, "estimated_charges": estimated_charges,
            "max_charges": data["max_charges"], "is_full": is_full,
        })
    return statuses

def run_single_profile_script(profile_name: str, templates_to_fix_paths: List[Path]):
    """Runs the single-profile painter script."""
    template_path_strs = [str(p.absolute()) for p in templates_to_fix_paths]
    command = [
        "python", "-m", "wplace_bot.run_single_profile", profile_name,
        *template_path_strs, # Unpack the list of paths as arguments
        "--profiles_folder", str(PROFILES_FOLDER),
    ]
    if templates_to_fix_paths:
        logging.info(f"Starting subprocess for profile: '{profile_name}' to fix {len(templates_to_fix_paths)} template(s).")
    else:
        logging.info(f"Starting subprocess for profile: '{profile_name}' for a charge-burn run.")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=PROFILE_TIMEOUT_SECONDS)
        
        if process.returncode != 0:
            logging.error(f"Subprocess for '{profile_name}' failed. Stderr:\n{stderr}")
        else:
            logging.info(f"Subprocess for '{profile_name}' finished. Stdout:\n{stdout}")
            
    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout exceeded for '{profile_name}'. Terminating...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    except Exception as e:
        logging.critical(f"Error running subprocess for '{profile_name}': {e}", exc_info=True)

def main_loop():
    """Main loop to select and run profiles."""
    global templates_to_fix

    logging.info("--- Manager starting main operational loop ---")
    
    while True:
        logging.info("--- Starting new manager cycle ---")
        
        profile_data = load_profile_data()
        statuses = calculate_profile_statuses(profile_data, PROFILES_TO_USE)
        
        full_profiles = [p for p in statuses if p["is_full"]]
        
        if not full_profiles:
            logging.info(f"No profiles are at full charges. Checking again in {IDLE_CHECK_INTERVAL_SECONDS}s.")
            time.sleep(IDLE_CHECK_INTERVAL_SECONDS)  # Short wait if idle
            continue

        # If we have profiles at full charge, run them all.
        logging.info(f"Found {len(full_profiles)} profile(s) at full charge. Preparing to run them.")
        
        with state_lock:
            templates_for_run = list(templates_to_fix)

        for profile_to_run in full_profiles:
            profile_to_run_name = profile_to_run['name']
            logging.info(f"Dispatching profile: '{profile_to_run_name}'")
            run_single_profile_script(profile_to_run_name, templates_for_run)
            time.sleep(5)  # Small delay between starting profiles
        
        logging.info(f"--- All full profiles have been dispatched. Waiting for {RUN_INTERVAL_SECONDS}s before next major cycle. ---")
        time.sleep(RUN_INTERVAL_SECONDS)


if __name__ == "__main__":
    all_template_paths = get_template_files(TEMPLATE_DIRECTORY)
    if not all_template_paths:
        logging.warning(f"No templates found in '{TEMPLATE_DIRECTORY}'. The bot will run in charge-burn only mode.")
    
    monitor_thread = threading.Thread(target=canvas_monitoring_worker, daemon=True)
    monitor_thread.start()
    
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("Manager script interrupted by user.")