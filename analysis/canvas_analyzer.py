# wplace_bot/analysis/canvas_analyzer.py
import functools
import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import requests
from PIL import Image
from stealth_requests import StealthSession

from wplace_bot.analysis.burn_strategies import STRATEGY_MAPPING
from wplace_bot.config import BURN_STRATEGIES, BURN_STRATEGY_CONFIG
from wplace_bot.utils.color_helper import find_color_ids_with_alpha

class CanvasAnalyzer:
    """Analyzes differences between local templates and the live canvas."""

    def __init__(self):
        self.pixel_queue: List[Tuple[float, int, int, int]] = []
        self.available_transparent_pixels: List[Tuple[int, int]] = []
        self.burn_candidates: List[List[Tuple[int, int]]] = []

    def needs_fixing(self) -> bool:
        """Returns True if the analysis found any pixels to paint."""
        return len(self.pixel_queue) > 0

    def analyze_templates(self, template_files: List[Path]):
        """
        Processes a list of template files to find pixels that need painting.
        
        This method populates the internal pixel_queue and available_transparent_pixels.
        """
        for template_path in template_files:
            try:
                template_data = self._get_template_data(template_path)
                self._analyze_single_template(template_data)
            except ValueError as e:
                logging.error(f"Skipping template due to error: {e}")
        
        self.pixel_queue.sort(key=lambda item: item[0], reverse=True)
        if len(template_files) > 1:
            logging.info(f"Finished all analysis. Total queue: {len(self.pixel_queue)} pixels.")

    def analyze_burn_candidates(self):
        """
        Initializes and runs all selected burn strategies from the config,
        preserving the order of candidates from each strategy.
        """
        if not BURN_STRATEGIES:
            logging.info("No burn strategies listed in config. Skipping analysis.")
            return

        self.burn_candidates = []
        total_candidates = 0
        with StealthSession() as session:
            for strategy_name in BURN_STRATEGIES:
                if strategy_name not in STRATEGY_MAPPING:
                    logging.warning(f"Burn strategy '{strategy_name}' not found in STRATEGY_MAPPING. Skipping.")
                    continue

                strategy_class = STRATEGY_MAPPING[strategy_name]
                strategy_config = BURN_STRATEGY_CONFIG.get(strategy_name, {})
                
                try:
                    logging.info(f"Executing burn strategy: '{strategy_name}'")
                    strategy = strategy_class(**strategy_config)
                    candidates = strategy.analyze(session)
                    if candidates:
                        # Add the list of candidates from this strategy as a new element
                        self.burn_candidates.append(candidates)
                        total_candidates += len(candidates)
                        logging.info(f"Strategy '{strategy_name}' found {len(candidates)} candidates.")
                except Exception as e:
                    logging.error(f"Error executing burn strategy '{strategy_name}': {e}", exc_info=True)
        
        logging.info(f"Finished all burn analyses. Total candidates found: {total_candidates} across {len(self.burn_candidates)} strategies.")


    def _get_template_data(self, template_path: Path) -> Dict[str, Any]:
        logging.info(f"Loading template from {template_path}...")
        match = re.match(r"(\d+),(\d+),(\d+),(\d+)\.png", template_path.name)
        if not match:
            raise ValueError(f"Template filename '{template_path.name}' must be 'tileX,tileY,pixelX,pixelY.png'")
        
        tx_start, ty_start, px_start, py_start = map(int, match.groups())
        img = Image.open(template_path).convert("RGBA")
        width, height = img.size
        
        covered_tiles = {
            (x, y)
            for x in range(tx_start, (tx_start * 1000 + px_start + width + 999) // 1000)
            for y in range(ty_start, (ty_start * 1000 + py_start + height + 999) // 1000)
        }
        
        logging.info(f"Template loaded: {width}x{height} pixels, covering {len(covered_tiles)} tiles.")
        return {
            "image": img, "path": template_path, "tx_start": tx_start, "ty_start": ty_start,
            "px_start": px_start, "py_start": py_start, "width": width, "height": height,
            "covered_tiles": covered_tiles
        }

    @staticmethod
    def _fetch_tile_image(tile_coords: Tuple[int, int], session: StealthSession) -> Tuple[Tuple[int, int], Image.Image | None]:
        tx, ty = tile_coords
        url = f"https://backend.wplace.live/files/s0/tiles/{tx}/{ty}.png"
        try:
            response = session.get(url, timeout=10, retry=3)
            response.raise_for_status()
            return tile_coords, Image.open(io.BytesIO(response.content)).convert("RGBA")
        except requests.RequestException as e:
            logging.error(f"Failed to fetch tile ({tx},{ty}): {e}")
            return tile_coords, None

    def _analyze_single_template(self, template_data: Dict[str, Any]):
        logging.info(f"Analyzing differences for template: {template_data['path'].name}...")
        live_tiles: Dict[Tuple[int, int], Image.Image] = {}

        with StealthSession() as session:
            with ThreadPoolExecutor(max_workers=10) as executor:
                fetch_func = functools.partial(self._fetch_tile_image, session=session)
                for coords, img in executor.map(fetch_func, template_data["covered_tiles"]):
                    if img: live_tiles[coords] = img

        newly_found_pixels, newly_found_transparent = [], []
        template_gx_start = template_data["tx_start"] * 1000 + template_data["px_start"]
        template_gy_start = template_data["ty_start"] * 1000 + template_data["py_start"]

        for (tx, ty), live_tile_img in live_tiles.items():
            tile_gx_start, tile_gy_start = tx * 1000, ty * 1000
            inter_x_start = max(template_gx_start, tile_gx_start)
            inter_y_start = max(template_gy_start, tile_gy_start)
            inter_x_end = min(template_gx_start + template_data["width"], tile_gx_start + 1000)
            inter_y_end = min(template_gy_start + template_data["height"], tile_gy_start + 1000)
            
            inter_width = inter_x_end - inter_x_start
            inter_height = inter_y_end - inter_y_start
            if inter_width <= 0 or inter_height <= 0: continue
            
            template_crop = (inter_x_start - template_gx_start, inter_y_start - template_gy_start, inter_x_start - template_gx_start + inter_width, inter_y_start - template_gy_start + inter_height)
            live_crop = (inter_x_start - tile_gx_start, inter_y_start - tile_gy_start, inter_x_start - tile_gx_start + inter_width, inter_y_start - tile_gy_start + inter_height)
            
            template_np = np.array(template_data["image"].crop(template_crop))
            live_np = np.array(live_tile_img.crop(live_crop))
            template_color_ids = find_color_ids_with_alpha(template_np)
            live_color_ids = find_color_ids_with_alpha(live_np)

            for py in range(inter_height):
                for px in range(inter_width):
                    global_x, global_y = inter_x_start + px, inter_y_start + py
                    template_id, live_id = template_color_ids[py, px], live_color_ids[py, px]
                    
                    if template_np[py, px, 3] > 128 and template_id != live_id:
                        newly_found_pixels.append((1.0, global_x, global_y, template_id))
                    elif live_id == 0: 
                        newly_found_transparent.append((global_x, global_y))
        
        self.pixel_queue.extend(newly_found_pixels)
        self.available_transparent_pixels.extend(newly_found_transparent)
        logging.info(f"Analysis for '{template_data['path'].name}' added {len(newly_found_pixels)} pixels.")
