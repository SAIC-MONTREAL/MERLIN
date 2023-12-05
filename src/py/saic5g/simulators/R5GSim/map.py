from .geometry import *
from .basestation import *
import numpy as np

class Map:
    """ 
    The Map class is responsible of initializing the scenario in the spatial domain (base stations, sectors) 
    """
    def __init__(self, shape, radius):
        """ 
        Args:
            shape (string): 'circle' or 'square'
            radius (int): map radius if circle, or 0.5 map side length if square in meters
        """
        self.shape = shape
        self.radius = radius
        self.geom = None


    def generate_map(self, site_radius, ue_density, ue_max_velocity, config_file):
        """ 
        Generate the map.

        Args:
            site radius (int): site coverage radius
            ue_density (int): average number of UEs withing a site

        Returns:
            geom (Geometry): An object that carries all the spatial information about the sites and the UEs
            site_list (list): A list of the site objects

        """
        self.geom = Geometry(self.radius, site_radius)
        n_sites, sites_positions, site_adj_m, site_relative_thetas = self.geom.generate_hexs(False, self.shape)
        site_list = []
        for i in range(n_sites):
            site_list.append(BaseStation(self.geom, i, sites_positions[i], site_adj_m[i], site_relative_thetas[i], config_file))
            
        self.geom.set_site_list(site_list)
        ue_assignments = self.geom.distribute_UEs(ue_density, ue_max_velocity)

        for i in range(n_sites):
            site_list[i].set_UE_info(np.where(ue_assignments==i)[0])

        return self.geom, site_list

    def update(self, t):
        """ 
        Update all the information of the sites, sectors and bands.
        
        Args:
            t: (int) timestep
        """
        for i, c in enumerate(self.geom.site_list):
            c.update_UE_info(np.where(self.geom.ue_assignments==i)[0], t)

    
