from .map import *
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
from .traffic import *


class RANSimulator5g:
    """
    The main simulation class that encapsulates everything.
    """
    def __init__(self, configs, map_shape):
        """ 
        Args:
            map_shape (string): 'circle' or 'square'
            map_r (int): map radius if circle, or 0.5 map side length if square in meters
            site_rad (int): site coverage in meters
            ue_density (int): average number of UEs withing a site
            ue_velocity (int): UEs movement velocity in m/s
        """
        
        
        self.config_file = configs
        self.map_shape = map_shape
        self.map_r = self.config_file.map_conf['map_r']
        self.site_rad = self.config_file.map_conf['site_rad']
        assert self.site_rad <= self.map_r/2, 'Dimensions error'

        self.ue_density = self.config_file.ue_conf['ue_density']
        self.ue_max_velocity = self.config_file.ue_conf['ue_max_velocity']
        self.time_step = 0
    
    def create_simulation(self):
        """ 
        Create a new simulation scenario as follows:
            1- Create a map
            2- Generate the sites
            3- Create the traffic and assign it to sites
        """
        self.map = Map(self.map_shape, self.map_r)
        self.geo, self.sites = self.map.generate_map(self.site_rad, self.ue_density, self.ue_max_velocity, self.config_file)
        self.traffic = Traffic(self.geo.n_UEs, self.config_file.traffic)
        for c in self.sites:
            c.set_UE_traffic(self.traffic)
        self.map.update(0)
        self.time_step +=1


    def simulate(self):
        """ 
        A simulation step (1 second) in the environment, detailed as follows:
            1- Schedule the users in each site
            2- Move the UEs in the environment
            3- Update UEs information, and remove UEs that are outside the coverage area
            4- Update the traffic and the map (updating the map means updating the sites, sectors and bands)
        """
        self.traffic.re_zero_throughput()
        for c in self.sites:
            c.schedule_users()
        self.geo.move_UEs(self.time_step)
        to_delete_ind = self.geo.update_UEs()
        self.traffic.update_UEs(to_delete_ind)
        self.traffic.generate_UE_demands()
        self.map.update(self.time_step)
        self.time_step +=1

