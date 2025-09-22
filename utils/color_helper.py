# wplace_bot/utils/color_helper.py
import numpy as np
from wplace_bot.config import COLOR_PALETTE_JSON

# Pre-process the palette for performance
COLOR_PALETTE = [c['rgb'] for c in COLOR_PALETTE_JSON]
FULL_PALETTE_NP = np.array(COLOR_PALETTE, dtype=np.uint8)

def find_color_ids_with_alpha(rgba_array: np.ndarray) -> np.ndarray:
    """
    Finds the closest color ID for each pixel in an RGBA array, correctly
    distinguishing between "Transparent" (0) and "Black" (1).
    """
    if rgba_array.ndim != 3 or rgba_array.shape[2] != 4:
        raise ValueError("Input array must be a 3D RGBA numpy array.")

    rgb_pixels = rgba_array[:, :, :3].reshape(-1, 3)
    opaque_palette = FULL_PALETTE_NP[1:]

    diff = rgb_pixels.astype(np.int64)[:, np.newaxis, :] - opaque_palette.astype(np.int64)[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    color_ids_flat = np.argmin(dist_sq, axis=1) + 1

    color_ids = color_ids_flat.reshape(rgba_array.shape[0], rgba_array.shape[1])
    color_ids[rgba_array[:, :, 3] < 128] = 0
    
    return color_ids