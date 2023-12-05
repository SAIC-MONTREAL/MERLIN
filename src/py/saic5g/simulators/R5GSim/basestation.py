import numpy as np
from .sector import *
import copy


class BaseStation:
    """
    A base station class as the first level of control hierarchy in the simulation. A base station is in the center of a site.
    This can be considered as macro or micro base station.
    """
    def __init__(self, geom, ID, coordinates, adj_BS, relative_BS_thetas, configs):
        """ 
        Args:
            geom (Geometry):  An object that carries all the spatial information about the sites and the UEs
            ID (int):  site ID
            coordinates (tuple): the basestation (x,y) coordinates
            adj_BS (list): a list of adjacent basestations IDs
            relative_BS_thetas (np.array):  (n_sites,) the relative angles between this basestation and the others
            configs (string): the configurations dictionary
        """
        self.geom = geom
        self.ID = ID
        self.coordinates = coordinates
        self.adj_BS =  adj_BS
        self.relative_BS_thetas = relative_BS_thetas
        self.configs = configs
        self.load_conf(self.configs.bs)
        self.n_UEs = None            # (int) number of UEs associated with this basestation
        self.ue_coordinates = None   # (np.array) (n_UEs, 2) UEs (x,y) coordinates
        self.ue_distances = None     # (np.array) (n_UEs, n_sites) UEs' d3 distances from each site
        self.ue_ind = None           # (list) list of associated UEs indices
        self.ue_thetas = None        # (np.array) (n_UEs,) the relative angle of each UE
        self.traffic = None          # (Traffic) a traffic object
        self.sectors_list = None     # (list) a list of Sector objects
        self.sectors_UE_ind = None   # (list) a list of lists which contatins the UEs indices that are association with each sector
        self.ue_sector_ID = {}       # (dict) The sector ID for each UE index.
        self.sector_adj_info = None  # (dict) a dictionary of lists that carry the information for adjacent sectors in each adjacent site
        
    
    def load_conf(self, conf):
        """
        load the configuration dict.

        Args:
            configs (string): the configurations dictionary
        """
        self.height = conf["height"]
        self.n_sectors = conf["n_sectors"]

    
    def set_UE_info(self, ue_ind):
        """
        Set the UEs info from the geom object

        Args:
            ue_ind (list): list of associated UEs indices
        """
        self.ue_ind = ue_ind
        self.n_UEs = len(self.ue_ind)
        self.ue_coordinates = self.geom.ue_coordinates[self.ue_ind]
        self.ue_distances = self.geom.d3_matrix[self.ue_ind, self.ID]
        self.update_UEs_thetas()
        self.assign_sector_UE()
        self.get_adj_sectors_info()
        self.create_sectors()

            
    
    def update_UE_info(self, ue_ind, t):
        """
        Update the UEs indices and info from the geom object after the new BS-UE re-assignment.
        Update the sector UEs indices.

        Args:
            ue_ind (list):  list of associated UEs indices
            t (int):  timestep
        """
        self.ue_ind = ue_ind
        self.n_UEs = len(self.ue_ind)
        self.ue_coordinates = self.geom.ue_coordinates[self.ue_ind]
        self.ue_distances = self.geom.d3_matrix[self.ue_ind, self.ID]
        self.update_UEs_thetas()
        self.assign_sector_UE()
        for i, sec in enumerate(self.sectors_list):
            sec.update_UE_info(self.sectors_UE_ind[i], t)

    
  

    
    def create_sectors(self):
        """
        Creates a Sectors list.
        Returns:
            self.sectors_list (list):  a list of Sector objects
        """
        sectors_list = []
        for i in range(self.n_sectors):
            sectors_list.append(Sector(self.geom, i, self.sector_adj_info[i], self.ID, self.sectors_UE_ind[i], self.traffic, self.configs))
        self.sectors_list = sectors_list
        return self.sectors_list

    
    def set_UE_traffic(self, traffic):
        """
        Set the UE traffic info from the traffic object to each sector.

        Args:
            traffic (Traffic): a traffic object
        """
        self.traffic = traffic
        for sector in self.sectors_list:
            sector.set_UE_traffic(traffic)
        

    def update_UEs_thetas(self):
        """
        Update the UEs angles, the relative angles between the UE and the base station.
        
        Returns:
            self.ue_thetas (np.array):  (n_UEs,) the relative angle of each UE.
        """
        centralized_UE_coordinates = copy.deepcopy(self.ue_coordinates)
        centralized_UE_coordinates[:,0], centralized_UE_coordinates[:,1] = centralized_UE_coordinates[:,0]-self.coordinates[0], centralized_UE_coordinates[:,1]-self.coordinates[1]
        self.ue_thetas = 360 + np.arctan2(centralized_UE_coordinates[:,1], centralized_UE_coordinates[:,0]) * 180 / np.pi
        self.ue_thetas[self.ue_thetas> 360] = self.ue_thetas[self.ue_thetas> 360] - 360
        return self.ue_thetas.astype(int)
    
    def assign_sector_UE(self):
        """
        Assigned UEs to a sector 
        All sectors starts from 270 degrees or 3*pi/2 rad, i.e.:
                              /-\                                 
                            /-   -\                               
                         /--       --\                            
                       /-      1      -\                          
                     -----          ------                        
                     |    \--- ----/     |                        
                     |         |         |                        
                     |    2    |   0     |                        
                     --        |        --                        
                     \-        |       -/                          
                       \--     |     --/                            
                         \-    |    -/ 
                           \-  |  -/                             
                              \-/                                 

        Returns:
            self.sectors_UE_ind (list):  a list of lists which contatins the UEs indices that are association with each sector
        """
        # 
        self.sectors_UE_ind = []
        if self.n_sectors == 1:
            self.sectors_UE_ind = self.ue_ind
        else:
            sector_thetas_start, sector_thetas_end = self.get_thetas_conf()
            dummy_ue_thetas = copy.deepcopy(self.ue_thetas)
            dummy_ue_thetas = dummy_ue_thetas + 90
            dummy_ue_thetas[dummy_ue_thetas>360] = dummy_ue_thetas[dummy_ue_thetas>360] - 360
            for i in range(self.n_sectors):
                ue_ind_temp = np.where( np.logical_and(dummy_ue_thetas>=sector_thetas_start[i],   dummy_ue_thetas<sector_thetas_end[i]) )[0]
                self.sectors_UE_ind.append(self.ue_ind[ue_ind_temp])
                self.geom.ues_sector_ID[self.ue_ind[ue_ind_temp]] = int(i)

        return self.sectors_UE_ind
    
    def get_thetas_conf(self): # rename to thetas sector
        """
        Get the sector starting and ending angles

        Returns:
            sector_thetas_start (list): the starting angle of each sector
            sector_thetas_end (list): the ending angle of each sector
        """
        sector_thetas_range = 360/self.n_sectors
        sector_thetas_start = np.array( [i*sector_thetas_range for i in range(self.n_sectors)] )
        sector_thetas_start[sector_thetas_start>360] = sector_thetas_start[sector_thetas_start>360] - 360

        sector_thetas_end =  np.array(  [(i+1)*sector_thetas_range for i in range(self.n_sectors)] )
        sector_thetas_end[sector_thetas_end>360] = sector_thetas_end[sector_thetas_end>360] - 360
        return sector_thetas_start, sector_thetas_end

    
    def get_adj_sectors_info(self):
        """
        Get the sector adjacency info in each neighboring site for the SINR calculation

        Returns:
            self.sector_adj_info (dict): a dictionary of lists that carry the information for adjacent sectors in each adjacent site
        """
        sector_adj_info = {i:[] for i in range(self.n_sectors) }
        sector_adj_m, sector_thetas = self.geom.get_sector_adj_and_theta(self.n_sectors)
        adj_m_1 = np.where(self.adj_BS == 1)[0]
        thetas_dummy_constant = np.array([300, 0, 60, 120, 180, 240])
        for site_ID in adj_m_1:
            rel_adj_theta = round(self.relative_BS_thetas[site_ID])
            sector_ID = sector_thetas[rel_adj_theta]
            site_index = np.where(thetas_dummy_constant==rel_adj_theta)[0][0]
            adj_sector_ID = round(sector_adj_m[sector_ID, site_index])

            sector_adj_info[sector_ID].append((site_ID, adj_sector_ID))
        self.sector_adj_info = sector_adj_info
        return self.sector_adj_info # dict:{secID: [(site_ID, adj_sector_ID)]}

    
    def get_load(self):
        """
        Calculate the average load in the basestation.

        Returns:
            avg_load (float): the average load in the basestation
        """
        loads = []
        for sec in self.sectors_list:
            loads.append(sec.get_load())
        avg_load = np.mean(loads)
        return avg_load

    
    def schedule_users(self):
        """
        Schedule the users in each sector
        """
        for s in self.sectors_list:
            s.schedule_users()

    def get_throughput_info(self):
        """
        Get the throughput info from the traffic object

        Returns:
            sum_th (float): the total throughput in the basestation
            mean_th (float): the mean throughput in the basestation
        """
        sum_tput = np.sum(self.traffic.throughput[self.ue_ind])
        mean_tput = np.mean(self.traffic.throughput[self.ue_ind])
        return sum_tput, mean_tput


        




