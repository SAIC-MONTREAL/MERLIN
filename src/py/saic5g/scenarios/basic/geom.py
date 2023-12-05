"""
Generic functions for scenario geometry generation (UE and cell positions).

Unless explicity specified, none of these account for the z dimension.

TODO: improve docstrings in RandomWalk class
"""
import json
import math

import numpy as np

from saic5g.scenarios.interfaces import Geom, Generator

class PrecomputedGeom(Geom):
    """
    A geometry defined by a set of pre-computed UE waypoints.
    """

    def __init__(self, cell_pos, ue_pos, ts_s, bbox=None):
        """
        Args:
            cell_pos((n_cell, 3) np.array): x,y,z cell positions
            ue_pos((n_ts, n_ues, 3) np.array): x,y,z,t ue positions
            ts_s (float): time step in seconds
            bbox (tuple): TODO
        """
        self.n_ts = ue_pos.shape[0]
        self.ts = ts_s
        self.tf = ts_s * (self.n_ts - 1)
        self._cell_pos = cell_pos
        self.bbox = bbox
        self._ue_pos = ue_pos
        if bbox is not None:
            self._ue_pos = self.wrap_to_bbox(ue_pos, bbox)
        self._d, self._azimuth = self.vector_cell_ue_distance_azimuth(cell_pos, ue_pos)

    def _t_idx(self, t):
        return int(t/self.ts)

    def cell_pos(self):
        return self._cell_pos

    def ue_pos(self, t):
        return self._ue_pos[self._t_idx(t), :, :]

    def cell_ue_distance(self, t):
        return self._d[self._t_idx(t), :, :]

    def cell_ue_azimuth(self, t):
        return self._azimuth[self._t_idx(t), :, :]

    def end_time(self):
        return self.tf

    def done(self, t):
        return t >= self.end_time()

    @staticmethod
    def vector_cell_ue_distance_azimuth(cell_pos, ue_pos):
        """
        Compute distance and azimuth between each UE/cell pair.

        Args:
            cell_pos (np.Array): (n_cell, 3) array
            ue_pos (np.Array): (n_steps[optional], n_ues, 3) array
        
        Returns:
            distance (np.Array): (n_steps[optional], n_cells, n_ues) distance between
                cells and UEs
            azimuth (np.Array): (n_steps[optional], n_cells, n_ues) azimuth angle between
                cells and UEs
        """
        has_time_dim = len(ue_pos.shape) == 3
        if not has_time_dim:
            ue_pos = ue_pos[None,:]
        n_ts = ue_pos.shape[0]
        n_ues = ue_pos.shape[1]
        n_cell = cell_pos.shape[0]
        cell_idx, ue_idx = np.unravel_index(np.arange(n_cell*n_ues), (n_cell, n_ues))
        cell_pos = cell_pos[cell_idx, :]
        ue_pos = ue_pos[:, ue_idx, :]
        diff = ue_pos - cell_pos
        diff2 = diff * diff
        d2 = np.sum(diff2, axis=2)
        d = np.sqrt(d2)
        d = np.reshape(d, (n_ts, n_cell, n_ues))
        azimuth = np.arctan2(diff[:, :, 1], diff[:, :, 0])
        azimuth = np.reshape(azimuth, (n_ts, n_cell, n_ues))
        if has_time_dim:
            return d, azimuth
        return d[0], azimuth[0]

    @staticmethod
    def wrap_to_bbox(pos, bbox):
        """
        Wrap to 2D bounding box

        Args:
            pos (np.array): (n_steps[optional], n_ues, 2) array of xy positions.
            bbox (tuple): (xmin, ymin, xmax, ymax)

        Returns:
            np.array: array of xy positions with same dims as pos
        """
        t_dim = True
        if len(pos.shape) == 2:
            pos = np.expand_dims(pos, 0)
        dx = bbox[2] - bbox[0]
        dy = bbox[3] - bbox[1]
        # handle points to the left of bbox by shifting everything to the right
        # assuming nothing is more than 1e5 bboxes away to the left.
        pos[:,:,0] = np.mod(pos[:,:,0] - bbox[0] + 1e5*dx, dx) + bbox[0]
        pos[:,:,1] = np.mod(pos[:,:,1] - bbox[1] + 1e5*dy, dy) + bbox[1]
        if not t_dim:
            pos = pos[0,:,:] 
        return pos


class RandomWalk(Generator):
    """
    UEs are randomly (uniformly) intialized and then do a random walk.
    """

    def __init__(
            self,
            cell_pos,
            n_ues,
            tf_s=100,
            bounds=None,
            std_displacement_m=1.,
            ts_s=1.,
            ue_height_m=1.5):
        """
        Args:
            cell_pos (np.Array)
            n_ues (int)
            tf_s (float): end time of simulation
            bounds (tuple): (minx, miny, maxx, maxy) bounding box. If None, will be determined by
                cell positions.
            std_displacement_m (float): Standard deviation of the displacement in one time step.
            ts_s (float): time step in seconds
            ue_height_m (float)
        """
        self.n_cell = len(cell_pos)
        self.n_ues = n_ues
        self.ts_s = ts_s
        self.n_ts = int(tf_s / ts_s)
        self.bounds = bounds or get_bounds(cell_pos)
        self.ue_height_m = ue_height_m
        self.std_displacement_m = std_displacement_m
        self.cell_pos = cell_pos

    def gen(self):
        ue_pos = np.zeros((self.n_ts, self.n_ues, 3))
        ue_pos[:, :, 2] = self.ue_height_m
        ue_pos[0, :, 0] = self.rs.uniform(size=self.n_ues)*(self.bounds[2] -self.bounds[0]) + self.bounds[0]
        ue_pos[0, :, 1] = self.rs.uniform(size=self.n_ues)*(self.bounds[3] -self.bounds[1]) + self.bounds[1]
        ue_pos[1:, :, :2] = self.rs.normal(0, self.std_displacement_m, size=(self.n_ts-1, self.n_ues, 2))
        ue_pos[:, :, :2] = np.cumsum(ue_pos[:,:,:2], axis=0)
        # PrecomputedGeom will automatically wrap to the bbox
        out = PrecomputedGeom(self.cell_pos, ue_pos, self.ts_s, bbox=self.bounds)
        return out


def get_bounds(pos, expand_m=50):
    """
    Return a box that bounds the given list of points. The box is expanded by expand_m meters
    in every direction.

    Args:
        pos(np.Array): (n_points, 2)
        expand_m (float)
    """
    xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
    ymin, ymax = pos[:, 1].min(), pos[:, 1].max()
    return (xmin - expand_m, y_min - expand_m, xmax + expand_m, ymax + expand_m)