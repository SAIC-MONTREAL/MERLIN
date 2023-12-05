
import numpy as np
from .band import *
import copy


class Sector():
    """
    A sector class as the second level of control hierarchy in the simulation. A sector is an area in the base station
    that can be a hemi-circle, triangle...etc.
    """
    def __init__(self, geom, ID, sector_adj_info, site_ID, ue_ind, traffic, configs):
        """ 
        Args:
            geom (Geometry): An object that carries all the spatial information about the sites and the UEs
            ID (int): sector ID
            sector_adj_info (list): a list of tuples that carries the information for adjacent sectors in each adjacent site
            site_ID (int): site ID
            ue_ind (list): the list of UE indices in the sector
            traffic (Traffic): a traffic object
            configs (string): the configurations dictionary
        """
        
        self.geom = geom
        self.ID = ID
        self.sector_adj_info = sector_adj_info
        self.site_ID = site_ID
        self.ue_ind = ue_ind
        self.temp_ue_ind = copy.deepcopy(self.ue_ind)
        self.traffic = traffic
        self.configs = configs
        self.load_conf(self.configs)
        self.n_UEs = len(self.ue_ind)
        self.ue_band_ID = {}   # (dict) the band ID for each UE
        self.band_UE_ind = None  # (list) the list of UE indices in each band
        self.bands_list = None   # (list) A list of Band objects
        self.throughput = 0
        self.n_load_balancing = 0
        self.assign_ue_bands()
        self.create_bands()
        self.bands_loads = np.array([1 for i in range(self.n_bands)])
        self.bands_CCCV =  np.array([band.CCCV for band in self.bands_list])

        self.bands_A2_th  = [-6 for i in range(self.n_bands)]
        self.bands_A5_th1 = [-6 for i in range(self.n_bands)]
        self.bands_A5_th2 = [-3 for i in range(self.n_bands)]
        
        

    def load_conf(self, configs):
        """
        load the configurations

        Args:
            configs : the configurations
        """
        conf = configs.sector
        self.n_bands = conf["n_bands"]
        self.load_balancing_enabled = conf["load_balancing"]
        self.load_balancing_algo = conf["load_balancing_algo"] # 1 for CIO
        self.fairness_th = conf["fairness_th"]
        self.IMLB_enabled = conf["IMLB_enabled"]
        self.bands_conf = [configs.band_configs[k] for k in configs.band_configs]
        

    def update_UE_info(self, ue_ind, t):
        """
        Update the UEs info in terms of band assignments.

        Args:
            ue_ind (list): list of associated UEs indices
            t (int): timestep
        """
        self.ue_ind = ue_ind
        self.n_UEs = len(ue_ind)

        if set(self.ue_ind) == set(self.temp_ue_ind):
            pass
        else:
            removed_UEs = list((set(self.temp_ue_ind) ^ set(self.ue_ind)) & set(self.temp_ue_ind))
            n_removed_UEs = len(removed_UEs)
            for ind in removed_UEs:
                band_ID = self.ue_band_ID[ind]
                self.band_UE_ind[band_ID].remove(ind)
                del self.ue_band_ID[ind]
            
            added_UEs = list((set(self.temp_ue_ind) ^ set(self.ue_ind)) & set(self.ue_ind))
            n_added_UEs = len(added_UEs)

            if self.IMLB_enabled:
                # load_factors = np.exp(-self.bands_loads)/np.sum(np.exp(-self.bands_loads)) # Alternative option
                load_factors = np.maximum((1-self.bands_loads), 1e-5)*self.bands_CCCV
                load_factors = (load_factors)/np.sum(load_factors) # 1e-5 to avoid nan values by dividing by 0.
            else:
                load_factors = np.array([1/self.n_bands for i in range(self.n_bands)])

            choices = np.arange(self.n_bands)

            for ind in added_UEs:
                band_ID = np.random.choice(choices, p=load_factors)
                self.band_UE_ind[band_ID].append(ind)
                self.ue_band_ID[ind] = band_ID
                
            self.temp_ue_ind = copy.deepcopy(self.ue_ind)

        
        if self.load_balancing_enabled:

            for i, band in enumerate(self.bands_list):
                band.update_UE_info(self.band_UE_ind[i])

            if (t+1)%2 == 0:
                if self.load_balancing_algo == 1:
                    self.n_load_balancing+=1
                    self.load_balancing_RSRQ()
        
        for i, band in enumerate(self.bands_list):
            band.update_UE_info(self.band_UE_ind[i])

        self.bands_loads = np.array([b.get_load() for b in self.bands_list])
        


    def set_UE_traffic(self, traffic):
        """
        Set the UE traffic info from the traffic object.

        Args:
            traffic (Traffic): a traffic object
        """
        self.traffic = traffic
        for b in self.bands_list:
            b.traffic = self.traffic



    def create_bands(self):
        """
        Creates a Bands list

        Returns:
            self.bands_list (list): a list of Band objects
        """
        bands_list = []
        for i in range(self.n_bands):
            bands_list.append(Band(self.geom, i, self.site_ID, self.ID, self.band_UE_ind[i], self.sector_adj_info, self.traffic, self.bands_conf[i]))
        self.bands_list = bands_list
        return self.bands_list

    # TODO: Improve band assignment
    def assign_ue_bands(self):
        """
        Assign UEs to a band.

        Returns:
            self.band_UE_ind (list): a list of lists which contatins the UEs indices that are association with each band
        """
        bands = [[] for i in range(self.n_bands)]
        for ind in self.ue_ind:
            band_ID = np.random.randint(self.n_bands)
            self.ue_band_ID[ind] = band_ID
            bands[band_ID].append(ind)
        self.band_UE_ind = bands
        return self.band_UE_ind
    
    def coded_ue_band_assignments(self):
        """
        Calculate the one-hot encoded assignment for each UE

        Returns:
            coded_assignments (np.array): (n_UEs, n_bands) the matrix for one-hot encoded UE assignments
        """
        coded_assignments = np.zeros((self.n_UEs, self.n_bands))
        bands_IDs = [self.ue_band_ID[ind] for ind in self.ue_ind]
        for i,b in enumerate(bands_IDs):
            coded_assignments[i][b] = 1
        return coded_assignments

    def load_balancing_RSRQ(self):
        """
        Perform A2 A5 events-based load balancing, the target cell is the cell with the best channel quality

        Returns:
            self.band_UE_ind (list): the list of band ID for each UE
        """
        self.calc_rsrq_bands()
        
        for ue_id in self.ue_ind:
            serving_band_ID = self.ue_band_ID[ue_id]
            serving_band_rsrq = self.rsrq_ues_dB[ue_id][serving_band_ID]
            target_band_ID = np.argmax(self.rsrq_ues_dB[ue_id])
            target_band_rsrq = self.rsrq_ues_dB[ue_id][target_band_ID]
            if serving_band_ID != target_band_ID:
                if serving_band_rsrq < self.bands_A2_th[serving_band_ID]:  #A2 event
                    if serving_band_rsrq < self.bands_A5_th1[serving_band_ID] and target_band_rsrq > self.bands_A5_th2[target_band_ID]: # A5 event
                        self.band_UE_ind[serving_band_ID].remove(ue_id)
                        self.band_UE_ind[target_band_ID].append(ue_id)
                        self.ue_band_ID[ue_id] = target_band_ID
        return self.band_UE_ind


    def calc_rsrq_bands(self):
        """
        Calculate the rsrq of each UE from each band

        Returns:
            self.rsrq_ues_dB (dict):  A dictionary of rsrq values for each UE from each band
        """
        self.rsrq_ues_dB = {ind:[ 0 for i in range(self.n_bands) ] for ind in self.ue_ind} # (dict) A dictionary of rsrq values for each UE from each band
        for ue_ind in self.ue_ind:
            band_ID = self.ue_band_ID[ue_ind]
            ue_gain = self.bands_list[band_ID].ues_gain
            for i in range(self.n_bands):
                _, rxp = self.bands_list[i].calculate_rsrp_others_dB(ue_ind, ue_gain)
                rsrq_dB = self.bands_list[i].calculate_rsrq_dB(rxp, ue_ind)
                self.rsrq_ues_dB[ue_ind][i] = rsrq_dB
            
        return self.rsrq_ues_dB
    
    def get_load(self):
        """
        Calculate the average load in the sector

        Returns:
            avg_load (float): the average load in the sector
        """
        loads = []
        for band in self.bands_list:
            loads.append(band.get_load())
        avg_load = np.mean(loads)
        return avg_load

    
    def schedule_users(self):
        """
        Schedule the users in each band
        
        Returns:
            self.throughput (float): the average tput in the sector
        """
        throughput = []
        for b in self.bands_list:
            t = np.round( b.schedule_users(), 3)
            throughput.append(t)

        self.throughput = np.mean(throughput)
        return self.throughput
    
