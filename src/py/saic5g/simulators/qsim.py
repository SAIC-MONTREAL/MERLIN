"""
(Q)uick (Sim)ulator.

This simulator trades off fidelity for speed.
"""

import numpy as np

from saic5g.utils.radio import to_db

EPSILON = 1e-9


def packet_loss(sinr, demand, capacity, assignment, handover_losses):
    """
    Compute packet loss.

    Split the capacity of the cell among the connected UEs in proportion to their
    load demand. Then apply Shannon-Hartley theorem for each UE to find the
    maximum throughput for that UE, using the allocated fraction of cell
    capacity as the maximum channel bandwidth. Compute packet loss as
    (demand - max throughput)/demand.

    Args:
        demand (np.Array): (n_ues,), demand of each UE
        capacity (np.Array): (n_cells,), capacity of each cell
        assignment (np.Array): (n_ues,), assignment of UEs to cells
        handover_losses (np.Array): (n_ues,), any losses due to handovers of UEs
            from one cell to another

    returns:
        (n_ues) np.array: per-ue packet loss
        (n_cell) np.array: total load through each cell
        (n_cell) np.array: per-cell packet loss
    """
    n_cell, n_ues = sinr.shape
    loads = np.zeros(sinr.shape)
    ue_idx = np.arange(n_ues)
    cell_idx = assignment[ue_idx]
    loads[cell_idx, ue_idx] = demand
    cell_loads = np.sum(loads, axis=1)
    ue_allocated_bw = (capacity / (cell_loads + EPSILON))[:, None] * loads
    throughput_factor = np.minimum(1, np.log2(1 + sinr))

    max_throughput = ue_allocated_bw * throughput_factor
    loss = loads - max_throughput
    # make sure that handovers are stricly bad

    if isinstance(handover_losses, np.ndarray):
        assert (handover_losses >= 0).all()
    else:
        assert handover_losses >= 0
    loss[cell_idx, ue_idx] += loads[cell_idx, ue_idx] * handover_losses

    loss = np.maximum(loss, 0)
    # Avoiding the losses exceeding the demand. This can happen when
    # there are high handover losses
    loss[cell_idx, ue_idx] = np.minimum(loss[cell_idx, ue_idx], demand)

    ue_loss = np.sum(loss, axis=0)
    cell_loss = np.sum(loss, axis=1)

    return ue_loss / (demand + EPSILON), cell_loads, cell_loss / (cell_loads + EPSILON)


def rsrp_sinr_rssi_rsrq(t, radio, distance, azimuth):
    """
    Compute RSRP, SINR, RSSI, and RSRQ

    Calculations based on https://www.cablefree.net/wirelesstechnology/4glte/lte-rsrq-sinr/

    Args:
        t (float): simulation time
        radio (Radio)
        distance (np.Array): (n_cells, n_ues)
        azimuth (np.Array): (n_cells, n_ues

    Returns:
        4 x np.Array: (n_cells, n_ues)
    """
    n_cell, n_ues = distance.shape
    n_prb = radio.n_prb(t, n_cell)[:, None]
    interfnoise = radio.interference(t, n_cell, n_ues) + \
        radio.noise(t, n_cell, n_ues)
    activity_factor = radio.activity_factor(t, n_cell)[:, None]
    rsrp = radio.rsrp(t, distance, azimuth)

    cell_capacity = radio.cell_capacity(t, n_cell)

    signal = 12 * activity_factor * n_prb * rsrp
    rssi = signal + interfnoise
    sinr = signal / interfnoise
    rsrq = n_prb * (rsrp/rssi)

    return rsrp, sinr, rssi, rsrq


class QSim:

    def __init__(self, scenario, time_step_s=1.):
        """
        Args:
            scenario (Scenario)
            time_step_s (float)
        """
        self.ts = time_step_s
        self.scenario = scenario
        self.geom = scenario.geom
        self.radio = scenario.radio
        self.traffic = scenario.traffic
        self._last_assignment = None

    def _obs(self, assignment):
        """
        Compute observed state given an assignment of UEs to cells.

        RSSI, RSRQ, and SINR calculations are taken from:
        https://www.cablefree.net/wirelesstechnology/4glte/lte-rsrq-sinr/

        returns:
            dict: simulator state output as described by
                https://github.sec.samsung.net/STAR/SAIC-MON-5G-RL/blob/master/simulator_specification.md
        """
        ue_pos = self.geom.ue_pos(self.t)
        d = self.geom.cell_ue_distance(self.t)
        a = self.geom.cell_ue_azimuth(self.t)
        n_cell, n_ues = d.shape
        t = self.t
        handover_losses = self.radio.handover_drop_load(
            t, assignment, self._last_assignment)
        demand = self.traffic.ue_demand(t, n_ues)
        cell_capacity = self.radio.cell_capacity(t, n_cell)

        rsrp, sinr, rssi, rsrq = rsrp_sinr_rssi_rsrq(t, self.radio, d, a)

        ue_pl, cell_demand, cell_pl = packet_loss(sinr, demand, cell_capacity, assignment,
                                                  handover_losses)

        return {
            'UEs': {
                'position': ue_pos,
                'rsrp': to_db(rsrp),
                'rsrq': to_db(rsrq),
                'sinr': to_db(sinr),
                'rssi': to_db(rssi),
                'serving_cell': assignment,
                'packet_loss': ue_pl,
                'demand': demand
            },
            'cells': {
                'position': self.geom.cell_pos(),
                'demand': cell_demand,
                'packet_loss': cell_pl
            },
            'simulation_time': self.t,
            'done': self.t + self.ts > self.geom.end_time()
        }

    def reset(self):
        """
        Resets the simulations. 

        UEs are initially assigned to cells to based on the scenario.

        returns:
            dict: simulator state output as described by
                https://github.sec.samsung.net/STAR/SAIC-MON-5G-RL/blob/master/simulator_specification.md
        """
        assignment = self.scenario.initial_assignment()
        self.t = 0.
        cell_pos = self.geom.cell_pos()
        ue_pos = self.geom.ue_pos(0)

        if assignment is None:
            rsrp = self.radio.rsrp(0, self.geom.cell_ue_distance(0),
                                   self.geom.cell_ue_azimuth(0))
            self._last_assignment = np.argmax(rsrp, axis=0)
        else:
            self._last_assignment = assignment

        return self._obs(self._last_assignment)

    def step(self, assignment):
        """
        Takes a step given a UE/cell assignment.

        args:
            assignment (n_ues) np.array

        returns:
            dict: simulator state output as described by
                https://github.sec.samsung.net/STAR/SAIC-MON-5G-RL/blob/master/simulator_specification.md
        """
        assignment = np.array(assignment)
        self.t += self.ts
        obs = self._obs(assignment)
        self._last_assignment = assignment

        return obs


class MultiBandCellularNetworkSimulator:
    """
    A multi-band wireless cellular network simulator

    :param scenario: Scenario object
    :param time_step: episode time step in TTI
    :param min_sinr_db: the minimum SINR for a UE in dB
    :param max_sinr_db: the maximum SINR for a UE in dB
    """

    def __init__(self,
                 scenario,
                 time_step=1.,
                 min_sinr_db=-20,
                 max_sinr_db=30):

        self.ts = time_step  # each time step corresponds to 1 TTI = 1ms in LTE
        self.scenario = scenario
        self.geom = scenario.geom
        self.radio = scenario.radio
        self.traffic = scenario.traffic
        self._last_assignment = None

        self.min_sinr_db = -20
        self.max_sinr_db = 30

    def compute_sinr(self, distances, azimuth, assignments):
        """
        Computes sinr for each UE. It only consider interference on the same
        band

        """
        n_cells, n_ues = distances.shape
        n_sites = int(n_cells/self.radio.n_freq)

        rxpdBm, rsrpDbm = self.radio.rsrp(self.t, distances, azimuth)
        rxpLin = np.power(10, 0.1 * rxpdBm)  # in mW

        assignment_matrix = np.eye(n_cells)[assignments]
        assignment_matrix = assignment_matrix.T
        assignment_matrix = assignment_matrix.reshape(
            n_sites, self.radio.n_freq, n_ues)

        ue_sinrs = np.zeros((n_ues,))

        loading = self.get_loading().reshape((n_sites, self.radio.n_freq))

        for ue_index in range(n_ues):
            cell_index = assignments[ue_index]
            pcc_index = int(cell_index % self.radio.n_freq)
            site_index = int(cell_index//self.radio.n_freq)

            cell_rxp = rxpLin[site_index, pcc_index, ue_index]
            cell_signal = cell_rxp*(loading[site_index, pcc_index]-0.001)

            interference = rxpLin[:, pcc_index, ue_index]*loading[:, pcc_index]

            interference = np.sum(interference)

            totalbandwidth = self.radio.resource_blocks[pcc_index] * \
                15000*self.radio.sub_carriers_per_rb  # in Hz

            noise_dbm = -174 + 10 * \
                np.log10(totalbandwidth)
            noise_mW = np.power(10, 0.1*noise_dbm)

            dividor = interference - cell_signal + noise_mW

            sinr = cell_rxp / dividor

            # Clip the SINR to be between min and max SINR values
            clipped_sinr = np.power(10, 0.1*np.clip(10*np.log10(sinr),
                                                    self.min_sinr_db,
                                                    self.max_sinr_db))

            if np.isnan(clipped_sinr):
                raise ValueError(f"SINR cannot be negative {sinr}")
            ue_sinrs[ue_index] = clipped_sinr

        return ue_sinrs, rsrpDbm

    def get_loading(self):
        """
        Get the current load for each cell
        """

        return 1/self.radio.sub_carriers_per_rb \
            + (1-1/self.radio.sub_carriers_per_rb)*self.loads

    def schedule_ues(self, ue_sinrs, demands, assignments):
        """
        Schedule UEs after mlb and update the network KPIs

        """
        distances = self.geom.cell_ue_distance(self.t)
        n_cells, n_ues = distances.shape
        n_sites = int(n_cells/self.radio.n_freq)

        # We assume that resources are allocated equally between UEs
        assignment_matrix = np.eye(n_cells)[assignments]
        assignment_matrix = assignment_matrix.T
        assignment_matrix = assignment_matrix.reshape(
            n_sites, self.radio.n_freq, n_ues)

        ue_rates = np.zeros_like(ue_sinrs)
        rbLeft = np.tile(self.radio.resource_blocks, n_sites).reshape(
            n_sites, self.radio.n_freq)

        loads = np.zeros((n_sites, self.radio.n_freq))
        dropped_ues = np.zeros((n_sites, n_ues))
        cell_throughput = np.zeros((n_sites, self.radio.n_freq))
        ue_capacity = np.zeros_like(ue_sinrs)

        # TODO
        # Sort UEs according to their signal quality before allocating resources

        for ue_index in range(n_ues):
            cell_index = assignments[ue_index]
            pcc_index = int(cell_index % self.radio.n_freq)
            site_index = int(cell_index//self.radio.n_freq)

            rbs = self.radio.resource_blocks[pcc_index]
            bandwidth = rbs * 15000*self.radio.sub_carriers_per_rb

            # 1TTI = 1ms
            capacity = bandwidth*np.log2(1+ue_sinrs[ue_index])/1000  # bits
            capacity /= 1024  # in Kbit
            ue_capacity[ue_index] = capacity

            required_rbs = np.ceil(rbs*demands[ue_index]/capacity)
            requested_rbs = min(required_rbs, rbLeft[site_index, pcc_index])

            if requested_rbs == 0:
                # no resources are left, the UE will be dropped
                dropped_ues[site_index, ue_index] += 1

            prb = requested_rbs / rbs
            scheduled_transmission = min(capacity * prb, demands[ue_index])
            rbLeft[site_index, pcc_index] -= requested_rbs
            loads[site_index, pcc_index] += requested_rbs

            ue_rates[ue_index] = scheduled_transmission
            cell_throughput[site_index, pcc_index] += scheduled_transmission

        resourceBlocks = np.tile(self.radio.resource_blocks, n_sites).reshape(
            n_sites, self.radio.n_freq)
        loads = loads/resourceBlocks
        self.loads = loads.reshape([-1])

        return ue_rates, cell_throughput, loads, dropped_ues, ue_capacity

    def _obs(self, assignments):
        """
        Compute observed state given an assignment of UEs to cells.

        returns:
            dict: simulator state output
        """

        ue_pos = self.geom.ue_pos(self.t)
        distances = self.geom.cell_ue_distance(self.t)
        angles = self.geom.cell_ue_azimuth(self.t)
        n_cells, n_ues = distances.shape
        t = self.t

        demands = self.traffic.ue_demand(t, n_ues)

        ue_sinrs, rsrpDbm = self.compute_sinr(distances, angles, assignments)

        ue_rates, cell_throughput, loads, dropped_ues, ue_capacity = self.schedule_ues(
            ue_sinrs, demands, assignments)

        return {
            'UEs': {
                'position': ue_pos,
                'rsrp': rsrpDbm,
                'sinr': ue_sinrs,  # in linear scale
                'rate': ue_rates,
                'serving_cell': assignments,
                'demand': demands,
                'capacity': ue_capacity,
                'distances': distances
            },
            'cells': {
                'position': self.geom.cell_pos(),
                'loads': loads,
                'Tput': cell_throughput,
                'dropped': dropped_ues
            },
            'simulation_time': self.t,
            'done': self.t + self.ts > self.geom.end_time()
        }

    def reset(self):
        """
        Resets the simulation. 

        UEs are initially assigned to cells to based on their signal quality.
        The cells are assumed to have zero load

        returns:
            dict: simulator state output
        """
        self.t = 0.
        cell_pos = self.geom.cell_pos()

        _, rsrp = self.radio.rsrp(0, self.geom.cell_ue_distance(0),
                                  self.geom.cell_ue_azimuth(0))

        rsrp = rsrp.reshape((-1, rsrp.shape[-1]))
        self._last_assignment = np.argmax(rsrp, axis=0)
        self.loads = np.zeros((cell_pos.shape[0]))

        return self._obs(self._last_assignment)

    def step(self, assignment):
        """
        Takes a step given a UE/cell assignment.

        args:
            assignment (n_ues) np.array

        returns:
            dict: simulator state output
        """
        assignment = np.array(assignment)
        self.t += self.ts
        obs = self._obs(assignment)
        self._last_assignment = assignment

        return obs
