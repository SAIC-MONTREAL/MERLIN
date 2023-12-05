
import numpy as np

class Traffic:
    """
    The Traffic class is responsible of initializing and generating the data traffic scenarios in the simulation.
    """ 
    def __init__(self, n_UEs, configs, time_step=0, MTU = 1500.0):
        """ 
        Args:
            n_UEs (int): number of UEs in the simulation
            time_step (int): simulation time resolution in seconds
            MTU (Maximum Transmisison Unit) (int): The maximum length of a data packet in bytes
        """
        self.n_UEs = n_UEs
        self.configs = configs
        self.MTU = MTU # Bytes
        self.time_step = time_step
        self.throughput = np.zeros(self.n_UEs)
        self.avg_packet_N = self.assign_avg_packet_N()
        self.n_times_scheduled = np.zeros(n_UEs)
        self.history_tput = np.ones(n_UEs)*.001
        self.ue_demands = np.zeros(n_UEs)
        self.ue_avg_demands = np.zeros(n_UEs)
        
        self.generate_UE_demands()
    
    def assign_avg_packet_N(self):
        """
        Assign UEs their average number of packets per second 
        Returns:
            avg_packet_N (np.array):  (n_UEs,) average number of packets per second for each UE
        """
        max_packet_n, min_packet_n = self.configs["max_packet_n"], self.configs["min_packet_n"]
        avg_packet_N = ( (np.random.rand(self.n_UEs)*(max_packet_n-min_packet_n) + min_packet_n) ).astype(int)
        return avg_packet_N
    
        
    # TODO: improve the traffic scenario
    def generate_UE_demands(self):
        """
        Assign UEs their accumelated demands in terms of number of Kbps

        Returns:
            incoming_demands (np.array): (n_UEs,) number of packets per second for each UE in Kb
        """
        n_packets_UEs = np.random.poisson(self.avg_packet_N)
        incoming_demands = self.MTU*n_packets_UEs*8.0/1024.0 # convert to Kb
        self.ue_demands += incoming_demands
        self.ue_demands = np.abs(self.ue_demands)
        new_avg_demands = self.ue_avg_demands + (incoming_demands- self.ue_avg_demands)/(self.time_step+1)
        self.ue_avg_demands = new_avg_demands
        self.time_step+=1
        return incoming_demands
    
    def re_zero_throughput(self):
        """
        Re-zero the instantenous tput for all the UEs
        """
        self.throughput = np.zeros(self.n_UEs)


    def update_UEs(self, remove_UEs_ind):
        """
        Update UEs info

        Args:
            remove_UEs_ind (list): the indices of the UEs that went out of coverage
        """
        self.ue_demands = np.delete(self.ue_demands, remove_UEs_ind)
        self.ue_avg_demands = np.delete(self.ue_avg_demands , remove_UEs_ind)
        self.avg_packet_N = np.delete(self.avg_packet_N, remove_UEs_ind)
        self.n_times_scheduled = np.delete(self.n_times_scheduled, remove_UEs_ind)
        self.history_tput = np.delete(self.history_tput, remove_UEs_ind)
        self.n_UEs = len(self.ue_demands)

