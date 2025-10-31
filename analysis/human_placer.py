# wplace_bot/analysis/human_placer.py
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist # No longer needed for pathing, but kept for potential future use

def get_human_like_ordered_pixels(candidate_pixels: List[Tuple[float, int, int, int]], max_pixels: int) -> List[Tuple[int, int, int]]:
    """
    Orders a list of candidate pixels in a human-like manner by prioritizing larger clusters.
    """
    if not candidate_pixels or max_pixels == 0:
        return []

    pixels_by_color = defaultdict(list)
    for _, gx, gy, cid in candidate_pixels:
        pixels_by_color[cid].append((gx, gy))

    all_clusters = []
    for cid, coords in pixels_by_color.items():
        if not coords: continue
        if len(coords) < 2:
            all_clusters.append({'cid': cid, 'coords': coords})
            continue

        coords_np = np.array(coords)
        db = DBSCAN(eps=2, min_samples=2).fit(coords_np)
        
        clusters_by_label = defaultdict(list)
        for i, label in enumerate(db.labels_):
            clusters_by_label[label].append(coords[i])
        
        for label, cluster_coords in clusters_by_label.items():
            if label == -1: # Noise points
                for coord in cluster_coords:
                    all_clusters.append({'cid': cid, 'coords': [coord]})
            else:
                all_clusters.append({'cid': cid, 'coords': cluster_coords})

    all_clusters.sort(key=lambda c: len(c['coords']), reverse=True)
    
    final_ordered_pixels = []
    for cluster in all_clusters:
        if len(final_ordered_pixels) >= max_pixels: break
        
        # --- START OF CHANGE ---
        # The O(N^2) nearest-neighbor pathfinding is replaced with a simple
        # and fast O(N log N) spatial sort.
        path = sorted(cluster['coords'], key=lambda p: (p[1], p[0])) # Sort by Y, then X
        # --- END OF CHANGE ---
        
        for p in path:
            if len(final_ordered_pixels) < max_pixels:
                final_ordered_pixels.append((p[0], p[1], cluster['cid']))
            else:
                break
                
    return final_ordered_pixels