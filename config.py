
# wplace_bot/config.py
import threading

import numpy as np

# --- General Configuration ---
LOG_FILE = "wplace_painter.log"
PROFILE_DATA_FILE = "profile_data.json"
DATA_LOCK = threading.Lock()


# --- New Manager Configuration ---
# Interval for the manager's background thread to check a random template.
RANDOM_TILE_CHECK_INTERVAL_SECONDS = 60

# --- Charge Burning Strategy ---
# A list of strategies to run sequentially for burning excess charges.
# An empty list disables strategic burning, falling back to random transparent pixels.
BURN_STRATEGIES = [
    "fixed_tile_burn",
    "enclosed_component"
]

# Configuration for specific strategies.
BURN_STRATEGY_CONFIG = {
    "enclosed_component": {
        "tile_x": 1188,
        "tile_y": 720,
        "color_id": 27,  # Corresponds to "Dark Pink"
    },
    "fixed_tile_burn": {
        "tile_x": 1662,
        "tile_y": 946,
    }
    # "another_strategy": { "some_param": "some_value" }
}

PRIORITY_FIX_THRESHOLD_PERCENT = 5.0

# The minimum number of charges a profile must have to be chosen for a priority run.
# This prevents wasting a run with a profile that can only place a few pixels.
PRIORITY_FIX_MIN_CHARGES = 100

# --- wplace.live Constants ---
COLOR_PALETTE_JSON = [
  {"name": "Transparent", "rgb": [0, 0, 0]},
  {"name": "Black", "rgb": [0, 0, 0]},{"name": "Dark Gray", "rgb": [60, 60, 60]},
  {"name": "Gray", "rgb": [120, 120, 120]},{"name": "Light Gray", "rgb": [210, 210, 210]},
  {"name": "White", "rgb": [255, 255, 255]},{"name": "Deep Red", "rgb": [96, 0, 24]},
  {"name": "Red", "rgb": [237, 28, 36]},{"name": "Orange", "rgb": [255, 127, 39]},
  {"name": "Gold", "rgb": [246, 170, 9]},{"name": "Yellow", "rgb": [249, 221, 59]},
  {"name": "Light Yellow", "rgb": [255, 250, 188]},{"name": "Dark Green", "rgb": [14, 185, 104]},
  {"name": "Green", "rgb": [19, 230, 123]},{"name": "Light Green", "rgb": [135, 255, 94]},
  {"name": "Dark Teal", "rgb": [12, 129, 110]},{"name": "Teal", "rgb": [16, 174, 166]},
  {"name": "Light Teal", "rgb": [19, 225, 190]},{"name": "Dark Blue", "rgb": [40, 80, 158]},
  {"name": "Blue", "rgb": [64, 147, 228]},{"name": "Cyan", "rgb": [96, 247, 242]},
  {"name": "Indigo", "rgb": [107, 80, 246]},{"name": "Light Indigo", "rgb": [153, 177, 251]},
  {"name": "Dark Purple", "rgb": [120, 12, 153]},{"name": "Purple", "rgb": [170, 56, 185]},
  {"name": "Light Purple", "rgb": [224, 159, 249]},{"name": "Dark Pink", "rgb": [203, 0, 122]},
  {"name": "Pink", "rgb": [236, 31, 128]},{"name": "Light Pink", "rgb": [243, 141, 169]},
  {"name": "Dark Brown", "rgb": [104, 70, 52]},{"name": "Brown", "rgb": [149, 104, 42]},
  {"name": "Beige", "rgb": [248, 178, 119]},{"name": "Medium Gray", "rgb": [170, 170, 170]},
  {"name": "Dark Red", "rgb": [165, 14, 30]},{"name": "Light Red", "rgb": [250, 128, 114]},
  {"name": "Dark Orange", "rgb": [228, 92, 26]},{"name": "Light Tan", "rgb": [214, 181, 148]},
  {"name": "Dark Goldenrod", "rgb": [156, 132, 49]},{"name": "Goldenrod", "rgb": [197, 173, 49]},
  {"name": "Light Goldenrod", "rgb": [232, 212, 95]},{"name": "Dark Olive", "rgb": [74, 107, 58]},
  {"name": "Olive", "rgb": [90, 148, 74]},{"name": "Light Olive", "rgb": [132, 197, 115]},
  {"name": "Dark Cyan", "rgb": [15, 121, 159]},{"name": "Light Cyan", "rgb": [187, 250, 242]},
  {"name": "Light Blue", "rgb": [125, 199, 255]},{"name": "Dark Indigo", "rgb": [77, 49, 184]},
  {"name": "Dark Slate Blue", "rgb": [74, 66, 132]},{"name": "Slate Blue", "rgb": [122, 113, 196]},
  {"name": "Light Slate Blue", "rgb": [181, 174, 241]},{"name": "Light Brown", "rgb": [219, 164, 99]},
  {"name": "Dark Beige", "rgb": [209, 128, 81]},{"name": "Light Beige", "rgb": [255, 197, 165]},
  {"name": "Dark Peach", "rgb": [155, 82, 73]},{"name": "Peach", "rgb": [209, 128, 120]},
  {"name": "Light Peach", "rgb": [250, 182, 164]},{"name": "Dark Tan", "rgb": [123, 99, 82]},
  {"name": "Tan", "rgb": [156, 132, 107]},{"name": "Dark Slate", "rgb": [51, 57, 65]},
  {"name": "Slate", "rgb": [109, 117, 141]},{"name": "Light Slate", "rgb": [179, 185, 209]},
  {"name": "Dark Stone", "rgb": [109, 100, 63]},{"name": "Stone", "rgb": [148, 140, 107]},
  {"name": "Light Stone", "rgb": [205, 197, 158]}
]
COLOR_PALETTE = [c['rgb'] for c in COLOR_PALETTE_JSON]
FULL_PALETTE_NP = np.array(COLOR_PALETTE, dtype=np.uint8)


# --- Profile Manager Configuration ---
# Folder where your Chrome profiles are stored.
PROFILES_FOLDER = "./Profiles"
# Directory to scan for .png templates.
TEMPLATE_DIRECTORY = "templates"

# List the names of the profile folders you want to run via the manager.
PROFILES_TO_USE = [
    "Profile 18",
    "Profile 17",
    "Profile 16",
    "Profile 15",
    "Profile 14",
    "Profile 13",
    "Profile 12",
    "Profile 11",
    "Profile 10",
    "Profile 9", 
    "Profile 8", 
    "Profile 7", 
    "Profile 6", 
    "Profile 5",
    "Profile 4", 
    "Profile 3", 
    "Profile 2",
    "Profile 1"
]

CHARGE_GAIN_RATE_SECONDS = 30  # 1 charge every 30 seconds
PROFILE_TIMEOUT_SECONDS = 120  # Max runtime for a single profile
RUN_INTERVAL_SECONDS = 300     # Time to wait between running profiles
IDLE_CHECK_INTERVAL_SECONDS = 60  # Wait time when no profiles are ready
