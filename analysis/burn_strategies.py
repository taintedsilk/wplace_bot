# wplace_bot/analysis/burn_strategies.py
import io
import logging
import random
from typing import List, Tuple, Set

import numpy as np
import requests
from PIL import Image
from stealth_requests import StealthSession

from wplace_bot.utils.color_helper import find_color_ids_with_alpha

class BurnStrategy:
    """Base class for all charge burning strategies."""
    def __init__(self, **kwargs):
        pass

    def analyze(self, session: StealthSession) -> List[Tuple[int, int]]:
        """
        Analyzes the canvas and returns a list of (gx, gy) coordinates to target.

        Args:
            session: An active StealthSession for making network requests.

        Returns:
            A list of (gx, gy) pixel coordinates.
        """
        raise NotImplementedError("Each strategy must implement the 'analyze' method.")


class EnclosedComponentBurnStrategy(BurnStrategy):
    """
    A strategy that finds connected components of a specific "sacrificial" color
    that are completely surrounded by transparent pixels on a designated tile.
    """
    def __init__(self, tile_x: int, tile_y: int, color_id: int, **kwargs):
        super().__init__(**kwargs)
        self.tile_x = tile_x
        self.tile_y = tile_y
        self.color_id = color_id
        logging.info(f"Initialized EnclosedComponentBurnStrategy for tile ({tile_x},{tile_y}) targeting color ID {color_id}.")

    @staticmethod
    def _fetch_tile_image(tx: int, ty: int, session: StealthSession) -> Image.Image | None:
        """Fetches and returns a single tile image."""
        url = f"https://backend.wplace.live/files/s0/tiles/{tx}/{ty}.png"
        try:
            response = session.get(url, timeout=10, retry=3)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGBA")
        except requests.RequestException as e:
            logging.error(f"Failed to fetch tile ({tx},{ty}) for burn analysis: {e}")
            return None

    def _find_enclosed_component(
        self,
        start_x: int,
        start_y: int,
        color_ids: np.ndarray,
        visited: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Finds a connected component of the target color using BFS and checks if it's enclosed.
        An enclosed component is one that is only connected to itself or to transparent pixels (ID 0).
        """
        height, width = color_ids.shape
        component = []
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))
        is_enclosed = True

        # BFS to find the component and check its boundary simultaneously
        head = 0
        while head < len(queue):
            x, y = queue[head]
            head += 1
            component.append((x, y))

            # Check all 8 neighbors in the 3x3 grid
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = x + dx, y + dy

                    # Check if neighbor is out of bounds; if so, we can't confirm it's enclosed
                    if not (0 <= ny < height and 0 <= nx < width):
                        is_enclosed = False
                        continue

                    neighbor_color = color_ids[ny, nx]

                    # If the neighbor is another color (not transparent and not our target color)
                    if neighbor_color != 0 and neighbor_color != self.color_id:
                        is_enclosed = False

                    # If the neighbor is part of our component and not yet visited, add to queue
                    elif neighbor_color == self.color_id and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        return component if is_enclosed else []

    def analyze(self, session: StealthSession) -> List[Tuple[int, int]]:
        """
        Analyzes the configured tile to find enclosed components suitable for burning.
        """
        logging.info(f"Executing burn analysis on tile ({self.tile_x}, {self.tile_y}).")
        tile_image = self._fetch_tile_image(self.tile_x, self.tile_y, session)

        if not tile_image:
            logging.error("Could not retrieve tile image; aborting burn analysis.")
            return []

        color_ids = find_color_ids_with_alpha(np.array(tile_image))
        height, width = color_ids.shape
        visited: Set[Tuple[int, int]] = set()
        all_candidates: List[Tuple[int, int]] = []

        # Iterate through every pixel to find potential starting points of components
        for y in range(height):
            for x in range(width):
                if color_ids[y, x] == self.color_id and (x, y) not in visited:
                    component = self._find_enclosed_component(x, y, color_ids, visited)
                    if component:
                        # Convert local (px, py) to global (gx, gy) coordinates
                        global_coords = [
                            (self.tile_x * 1000 + comp_x, self.tile_y * 1000 + comp_y)
                            for comp_x, comp_y in component
                        ]
                        all_candidates.extend(global_coords)

        logging.info(f"EnclosedComponentBurnStrategy found {len(all_candidates)} candidates.")
        return all_candidates



class RandomTileInRadiusBurnStrategy(BurnStrategy):
    """
    A strategy that targets all transparent pixels on a randomly selected tile
    within a specified radius of a center point.
    """
    def __init__(self, center_tile_x: int, center_tile_y: int, radius: int, **kwargs):
        super().__init__(**kwargs)
        self.center_x = center_tile_x
        self.center_y = center_tile_y
        self.radius = radius
        logging.info(f"Initialized RandomTileInRadiusBurnStrategy around ({self.center_x},{self.center_y}) with radius {self.radius}.")

    @staticmethod
    def _fetch_tile_image(tx: int, ty: int, session: StealthSession) -> Image.Image | None:
        """Fetches and returns a single tile image."""
        url = f"https://backend.wplace.live/files/s0/tiles/{tx}/{ty}.png"
        try:
            response = session.get(url, timeout=10, retry=3)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGBA")
        except requests.RequestException as e:
            logging.error(f"Failed to fetch tile ({tx},{ty}) for burn analysis: {e}")
            return None

    def _get_random_tile_in_radius(self) -> Tuple[int, int]:
        """Calculates and returns a random tile within the specified radius."""
        candidate_tiles = []
        radius_sq = self.radius ** 2
        
        # Iterate over the bounding box of the circle
        for tx in range(self.center_x - self.radius, self.center_x + self.radius + 1):
            for ty in range(self.center_y - self.radius, self.center_y + self.radius + 1):
                # Check if the tile is within the circle using squared Euclidean distance
                dist_sq = (tx - self.center_x)**2 + (ty - self.center_y)**2
                if dist_sq < radius_sq:
                    candidate_tiles.append((tx, ty))
        
        if not candidate_tiles:
            logging.warning("No candidate tiles found in radius. Defaulting to center tile.")
            return (self.center_x, self.center_y)
            
        return random.choice(candidate_tiles)

    def analyze(self, session: StealthSession) -> List[Tuple[int, int]]:
        """
        Analyzes a random tile within the radius to find all transparent pixels.
        """
        target_tile_x, target_tile_y = self._get_random_tile_in_radius()
        logging.info(f"Executing random tile burn analysis on tile ({target_tile_x}, {target_tile_y}).")
        tile_image = self._fetch_tile_image(target_tile_x, target_tile_y, session)

        if not tile_image:
            logging.error("Could not retrieve tile image; aborting burn analysis.")
            return []

        color_ids = find_color_ids_with_alpha(np.array(tile_image))
        
        # Find all coordinates where the color ID is 0 (Transparent)
        transparent_pixels = np.argwhere(color_ids == 0)
        
        # Convert local (py, px) to global (gx, gy) coordinates
        # Note: argwhere returns (row, col) which corresponds to (y, x)
        global_coords = [
            (target_tile_x * 1000 + px, target_tile_y * 1000 + py)
            for py, px in transparent_pixels
        ]
        
        logging.info(f"RandomTileInRadiusBurnStrategy found {len(global_coords)} transparent pixels on tile ({target_tile_x}, {target_tile_y}).")
        return global_coords


# --- Strategy Mapping ---
# To add a new strategy:
# 1. Create a new class that inherits from BurnStrategy.
# 2. Implement the analyze() method.
# 3. Add the class to this dictionary with a unique key.
STRATEGY_MAPPING = {
    "enclosed_component": EnclosedComponentBurnStrategy,
    "random_tile_in_radius_burn": RandomTileInRadiusBurnStrategy,
    # "another_strategy": AnotherStrategyClass,
}