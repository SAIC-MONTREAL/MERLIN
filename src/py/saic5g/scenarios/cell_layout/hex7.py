import math
import numpy as np


def hex7_cells(base_dist_m, base_height_m, cells_per_site=1):
    """
    Generate cell positions for 7 base stations arragned in a hexagonal pattern.
    
    Args:
        base_dist_m (float)
        base_height_m (float)
        cells_per_site (int)
        
    Returns:
        np.Array: (n_cells, 3)
    """
    bases = [
        (0, 0, base_height_m),
        (base_dist_m, 0, base_height_m),
        (base_dist_m/2, math.sqrt(3) * base_dist_m / 2, base_height_m),
        (-base_dist_m/2, math.sqrt(3) * base_dist_m / 2, base_height_m),
        (-base_dist_m, 0, base_height_m),
        (-base_dist_m/2, -math.sqrt(3) * base_dist_m / 2, base_height_m),
        (base_dist_m/2, -math.sqrt(3) * base_dist_m / 2, base_height_m),
    ]
    bases = np.array(bases)
    cells = np.repeat(bases, cells_per_site, axis=0)
    return cells

def hex7_bounds(base_dist_m):
    """
    Generate bounding box for 7 base stations arragned in a hexagonal pattern.
    
    Args:
        base_dist_m (float)
        
    Returns:
        tuple: (xmin, ymix, xmax, ymax) bounding box
    """
    bbox = (-base_dist_m*1.5, -base_dist_m*1.5, base_dist_m*1.5, base_dist_m*1.5)
    return bbox
