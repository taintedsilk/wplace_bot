# wplace_bot/analysis/burn_strategies.py
import io
import logging
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

        print(f"EnclosedComponentBurnStrategy found {len(all_candidates)} candidates.")
        return all_candidates


# --- Strategy Mapping ---
# To add a new strategy:
# 1. Create a new class that inherits from BurnStrategy.
# 2. Implement the analyze() method.
# 3. Add the class to this dictionary with a unique key.
STRATEGY_MAPPING = {
    "enclosed_component": EnclosedComponentBurnStrategy,
    # "another_strategy": AnotherStrategyClass,
}