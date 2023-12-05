import numpy as np
from . import schedulers as sched

class Band():
    """
    A band class as the third level of control hierarchy in the simulation. A band-sector pair is a cell.
    """
    def __init__(self, geom, ID, site_ID, sector_ID, ue_ind, sector_adj_info, traffic, configs):
        """ 
        Args:
            geom  (Geometry): An object that carries all the spatial information about the sites and the UEs
            ID (int): band ID
            ue_ind (list): the list of UE indices in the sector
            site_ID (int): site ID
            sector_adj_info (list): a list of tuples that carries the information for adjacent sectors in each adjacent site
            traffic (Traffic): a traffic object
            configs (dict): dictionary of configs
        """
        self.geom = geom
        self.ID = ID
        self.site_ID = site_ID
        self.sector_ID = sector_ID
        self.ue_ind = ue_ind
        self.sector_adj_info = sector_adj_info
        self.traffic = traffic
        self.n_UEs = len(self.ue_ind)
        self.ue_distances = self.geom.d3_matrix[self.ue_ind, self.site_ID] # (np.array) (n_UEs,) UEs' d3 distances
        self.load_conf(configs) # <-- check here the constants before you proceed
        self.sinr_values = np.array([-11, -7, -2, 3, 4, 7, 8, 10, 12, 15, 18, 19, 20, 22, 25]) # Table SINR values https://ieeexplore.ieee.org/document/9435258
        self.sinr_MCS_dict = {-11: 2, -7: 2, -2: 2, 3: 4, 4: 4, 7: 4, 8: 6, 10: 6, 12: 6, 15: 6, 18: 6, 19: 8, 20: 8, 22: 8, 25: 8} # MCS: Modulation coding scheme
        self.sinr_CQI_dict = {-11: 1, -7: 2, -2: 3, 3: 4, 4: 5, 7: 6, 8: 7, 10: 8, 12: 9, 15: 10, 18: 11, 19: 12, 20: 13, 22: 14, 25: 15} # CQI: Channel Quality Indicator
        self.sinr_max = 25
        self.sinr_min = -11
        self.is_mmWave = True if self.fc>=28 else False
        self.ues_gain = 2        # (int) UEs' antenna gain #TODO: change this to make it more realistic
        self.load = 0            # (int) Band load
        self.n_subcarriers_per_RB  = 12  # (int) number of subcarriers (SC) in a single PRB. It's fixed and always 12
        self.subframe_per_ms = int( 2**np.log2(self.sc_spacing/15) ) # (int) number of subframes within 1ms. It depends on the SC spacing.
        self.available_prbs = int(self.bw*1e3/(self.n_subcarriers_per_RB*self.sc_spacing)) # (int) number of PRBs in the whole band
        self.total_n_subcarriers = self.n_subcarriers_per_RB*self.available_prbs           # (int) number of total subcarriers in the whole band
        self.n_data_symbols = 11 # (int) number of data symbols in a PRB in time domain. In 5G, 14 symbols in total, but 11 for data only
        self.n_REs_prb = self.n_data_symbols*self.n_subcarriers_per_RB  # The number of resource element (time-frequency component) in a PRB. Each PRB is a matrix (n_data_symbols, n_subcarriers)
        self.DL_ratio = 3/4      # (int) DL/UL TDD ratio. In each second, an uplink synchronization and CSI signals are required
        self.ue_streams = 2      # (int) Number of data streams that a UE can handle simultaneously. UEs have 2 data streams on average.
        self.n_ue_served = int(self.tx_mimo/self.ue_streams)   # (int) How many UEs can be served simultaneously in a scheduling iteration
        self.simulation_res = 10 # (int) Simulation resolution in ms
        self.n_ue_served_per_s = int( (1000/self.simulation_res)*self.n_ue_served)
        self.slots_per_N_ms = self.simulation_res*self.subframe_per_ms # Number of time slots per 1 simulation iteration. E.g., 10ms/iteration x 2 subframes/ms = 20 subframes/iterations
        self.available_slots = self.DL_ratio*self.slots_per_N_ms  # Available DL time slots. E.g., 3/4 * 20 = 15 subframes/iterations
        self.available_prbs_total = int(self.available_slots*self.available_prbs*(1000/self.simulation_res)) # Total number of PRBs available in time-frequency domain within 1 second
        self.band_capacity_total = \
            (self.tx_mimo)*(2**6)*self.n_REs_prb*self.available_prbs_total/1024 # Band Capacity in KB per second
        
        self.noise = 10**((-174 + 10 * np.log10(self.bw*1e6)  - 30)/10) # (float) Thermal noise in the whole band
        self.throughput = 0
        self.t_c = 20 
        
        
    def load_conf(self, conf):
        """
        load the configurations
        Args:
            configs (dict): the configuration dict
        """
        self.fc = conf["cf"] # Central Frequency in  GHz
        self.bw = conf["bw"] # Total available bandwidth in MHz
        self.antenna_g = conf["antenna_gx"]  # Antenna gain in dBi
        self.antenna_p = conf["antenna_p"]   # Antenna transmission power in dB
        self.tx_mimo = conf["tx_mimo"]       # Number of antennas in MIMO
        self.sc_spacing = conf["sc_spacing"] # Subcarrier spacing in kHz
        self.CCCV = conf["CCCV"]             # Cell capacity class value 
        
        

    def get_load(self):
        """
        Calculate the load in the band.
        The load is defined as the demands/cell_capacity
        if load<1:
            the band is underutilized
        if load>1:
            the band is overutilized

        Returns:
            load (float): the load in the band
        """
        requested_throughput = np.sum(self.traffic.ue_demands[self.ue_ind])
        load = np.round(requested_throughput/self.band_capacity_total, 3)
        return load
    

    def update_UE_info(self, ue_ind):
        """
        Set the UE info in terms of assignments and distances.

        Args:
            ue_ind (list): list of associated UEs indices
        """
        self.ue_ind = np.array(ue_ind).tolist()
        self.n_UEs = len(ue_ind)
        self.ue_distances = self.geom.d3_matrix[self.ue_ind, self.site_ID]
        self.calculate_ues_rsrp_rsrq_dB()
        

    
    def calculate_sinr_dB(self, rxp, ue_ind):
        """
        Calculate the SINR in dB

        Args:
            rxp: (float) the received power value
            ue_ind (int): a UE index

        Returns:
            sinr_dB (float): the SINR value in dB
        """
        interference = self.calculate_interference(ue_ind)

        total_band_rxp = (self.available_prbs*self.n_subcarriers_per_RB)*rxp
        sinr_dB = 10*np.log10(total_band_rxp/(interference+self.noise))
        sinr_dB = np.clip(sinr_dB, self.sinr_min, self.sinr_max).astype(int)
        return sinr_dB


    def calculate_interference(self, ue_ind):
        """
        Calculate the interference in the whole band

        Args:
            ue_ind (int): a UE index

        Returns:
            inteference (float): the total interference power
        """
        interference = 0
        for (site_ID, sec_ID) in self.sector_adj_info:
            band_obj = self.geom.site_list[int(site_ID)].sectors_list[int(sec_ID)].bands_list[self.ID]
            _, rxp_other = band_obj.calculate_rsrp_others_dB(ue_ind, self.ues_gain)
            interference += rxp_other*min(band_obj.get_load(), 1)
        
        return interference

    
    def calculate_rsrq_dB(self, rxp, ue_ind):
        """
        Calculate the RSRQ in dB, considering the received power per PRB, activity factor of 1 always,
        the noise and intereference are for the whole band.

        Args:
            rxp (float): the received power value
            ue_ind (int): a UE index

        Returns:
            rsrq_dB (float): the RSRQ value in dB
        """
        interference = self.calculate_interference(ue_ind)
        rssi = min(self.get_load(), 1)*self.available_prbs*rxp + self.noise + interference
        rssi_dB = 10*np.log10(rssi)
        rssi_dBm = rssi_dB + 30

        rsrq = self.available_prbs*rxp/rssi
        rsrq_dB = 10*np.log10(rsrq)
        
        return rsrq_dB
    


    def calculate_ues_rsrp_rsrq_dB(self):
        """
        Calculate the rsrp and rsrq for all the UEs in this band.
        Refs:
        3GPP TR 38.901 version 15.0.0 Release 15
        https://ieeexplore.ieee.org/document/6831690

        Returns:
            self.band_rsrp (np.array): (n_UEs,) the rsrp values for all UEs
        """
        if self.is_mmWave:
            ue_pathloss = 60 + 20*np.log10(self.fc) + 4.51*10*np.log10(self.ue_distances)
        else:
            ue_pathloss = 32.54 + 20*np.log10(self.fc) + 30*np.log10(self.ue_distances)
        
        shadowing =  np.clip( np.random.lognormal(1, 7.8, self.n_UEs), 1, 30)
        rsrp_dB_total = self.antenna_g + self.antenna_p + self.ues_gain - ue_pathloss - shadowing
        
        self.band_rsrp = rsrp_dB_total - 10*np.log10(self.total_n_subcarriers)
        self.rxp = 10**(self.band_rsrp/10)

        rsrqs = []
        for ind,rxp in zip(self.ue_ind, self.rxp):
            rsrqs.append(self.calculate_rsrq_dB(rxp, ind))
        self.rsrq = np.array(rsrqs)
        print(self.rsrq)
        return self.band_rsrp
    

    def calculate_rsrp_others_dB(self, ue_ind, ue_gain):
        """
        Calculate the rsrp for another UE outside the band given the index and its gain.
        Refs:
        3GPP TR 38.901 version 15.0.0 Release 15
        https://ieeexplore.ieee.org/document/6831690

        Args:
            ue_ind (int): the UE index
            ue_gain (int): the UE antenna gain

        Returns:
            band_rsrp (float): the rsrp value
        """
        if self.is_mmWave:
            ue_pathloss = 60 + 20*np.log10(self.fc) + 4.51*10*np.log10(self.geom.d3_matrix[ue_ind, self.site_ID]) 
        else:
            ue_pathloss = 32.54 + 20*np.log10(self.fc) + 30*np.log10(self.geom.d3_matrix[ue_ind, self.site_ID]) 

        shadowing = np.clip( np.random.lognormal(1, 7.8), 1, 30)
        rsrp_dB_total = self.antenna_g + self.antenna_p + ue_gain - ue_pathloss - shadowing
        band_rsrp = rsrp_dB_total - 10*np.log10(self.total_n_subcarriers)
        rxp = 10**(rsrp_dB_total/10)
        return band_rsrp, rxp

    def map_sinr_mcs(self, sinr):
        """
        Map the SINR value to a MCS

        Args:
            sinr (float): the sinr value
        
        Returns:
            mcs (int): the modulation coding scheme
        """
        sinr_val = self.sinr_values.flat[np.abs(self.sinr_values - sinr).argmin()]
        mcs = self.sinr_MCS_dict[sinr_val]
        return mcs 
    
    def map_sinr_CQI(self, sinr):
        """
        Map the SINR value to a MCS

        Args:
            sinr (float): the sinr value
        
        Returns:
            cqi (int): the channel quality indicator
        """
        sinr_val = self.sinr_values.flat[np.abs(self.sinr_values - sinr).argmin()]
        cqi = self.sinr_CQI_dict[sinr_val]
        return cqi
    
    def schedule_users(self):
        """
        Perform the scheduling (time and resource allocation) for all UEs in the band

        Returns:
            tput (float): the average tput in this band
        """
        tput = sched.PFS(self)
        return tput


    
