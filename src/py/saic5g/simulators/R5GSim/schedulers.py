import numpy as np
import copy

def PFS(band):
    """
    Perform the scheduling (time and resource allocation) (we assume SC-FDMA) for all UEs in the band using 
    the Proportional Fairness Scheduler (PFS).

    Args:
        band (Band): the band object

    Returns:
        mean_tput (float): the average tput in this band after the scheduling in Kbps
    """
    # ue_ind [n_UEs] is global index for each UE
    
    hist_rates = copy.deepcopy(band.traffic.history_tput[band.ue_ind]) # [n_UEs]  rates
    ue_demands = copy.deepcopy(band.traffic.ue_demands[band.ue_ind])   # [n_UEs] demands per 1 stream
    tput = np.zeros_like(ue_demands)
    local_ind = np.arange(0, band.n_UEs)
    time_sched = np.zeros_like(ue_demands)
    ue_sinrs = []
    for i in range(band.n_UEs):
         ue_sinrs.append(band.calculate_sinr_dB(band.rxp[i], band.ue_ind[i]))
    ue_sinrs = np.array(ue_sinrs) # [n_UEs] sinr

    mcs = [band.map_sinr_mcs(sinr) for sinr in ue_sinrs]
    ue_CQI = np.array([band.map_sinr_CQI(sinr) for sinr in ue_sinrs])
    ue_BLER = np.random.chisquare(4.01-np.log2(ue_CQI))/100
    data_per_prb = np.power(2, mcs)*band.n_REs_prb


    prb_per_ue = ue_demands*1024//(band.ue_streams*data_per_prb*(1-ue_BLER)) # Requested
    prb_per_ue[prb_per_ue<1] = 0
    available_prbs_per_i = band.available_prbs*band.available_slots # Total prbs in freq-time domain per iteration
    prb_per_ue = np.minimum(prb_per_ue, available_prbs_per_i)
    rates = band.ue_streams*data_per_prb*prb_per_ue/1024
    PFS_factors = rates/hist_rates
    scheduling_iterations = 1000//band.simulation_res
    for i in range(scheduling_iterations): 
        max_PFS_factors_ind = np.argsort(-PFS_factors)[:band.n_ue_served]
        served_KB = rates[max_PFS_factors_ind]
        tput[max_PFS_factors_ind] += served_KB
        ue_demands[max_PFS_factors_ind] -= served_KB
        ue_demands[ue_demands<0] = 0
        time_sched[max_PFS_factors_ind] +=1
        not_sched_ind = list((set(max_PFS_factors_ind) ^ set(local_ind)) & set(local_ind))

        hist_rates[max_PFS_factors_ind] = (1-1/band.t_c)*hist_rates[max_PFS_factors_ind] + (1/band.t_c)*served_KB
        hist_rates[not_sched_ind] = (1-1/band.t_c)*hist_rates[not_sched_ind]

        prb_per_ue = ue_demands*1024//(band.ue_streams*data_per_prb*(1-ue_BLER))
        prb_per_ue[prb_per_ue<1] = 0

        if np.all(prb_per_ue<1):
            break

        prb_per_ue = np.minimum(prb_per_ue, available_prbs_per_i)
        rates = band.ue_streams*data_per_prb*prb_per_ue/1024
        PFS_factors = rates/hist_rates
        

    
    band.traffic.throughput[band.ue_ind] = tput
    band.traffic.ue_demands[band.ue_ind] = ue_demands
    band.traffic.history_tput[band.ue_ind] = hist_rates
    band.traffic.n_times_scheduled[band.ue_ind] += time_sched
    band.throughput = tput
    mean_tput = np.nan_to_num(np.mean(band.throughput))
    return mean_tput
