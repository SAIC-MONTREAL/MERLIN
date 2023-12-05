from gym import spaces
import numpy as np

from saic5g.envs.per_ue_base import PerUeHandover
from saic5g.simulators.qsim import MultiBandCellularNetworkSimulator


class A3BasedMLB(PerUeHandover):
    """
    Environment for mobility load balancing (MLB) based on A3 event. 

    In 4G LTE, the handover of a UE from a serving cell s to another cell t
    happens when this condition is satistified:

        RSRP_t + O_{t->s} >= RSRP_s + O_{s->t} + Hys

    where O_{t->s} is the cell individual offset (CIO) of cell t with respect
          to the cell s. The CIO value O_{t->s} can be interpreted as the offset 
          value that makes the measured RSRP of cell t stronger (or weaker) when 
          compared with the measured RSRP of cell s. 
          Hys is the hysteresis value used to minimize the probability of 
          occurrence of the ping-pong scenario

    This environment follows the same assumption as in [1] and assumes that
    the same offset value is used for all the neighbors (i.e., O_{i->j}= O_i 
    for all j in the i's neighbors). Thus, the handover condition can becomes :

        RSRP_t + O_t >= RSRP_s + O_s + Hys

    This simplification allows the number of CIOs to be equal to the number
    of cells, thus greatly reducing the search space.

     Action Space:
     The action is a `ndarray` with shape `(n_cells,)` which can take values 
     from 0 to N where N is the number of CIO levels in dB (e.g., 0 ==> -6 dB)

     Observation Space:
     By default, the observation is a `ndarray` with shape `(n_cells,)` with 
     the values corresponding to the PRB utilization ratio for each cell. 
     It is possible to add more features in the observation such as the cell 
     throughput and the UEs SINR. The table below summarizes the available 
     observation features::

        | Num | Observation           | Min                 | Max               |
        |-----|-----------------------|---------------------|-------------------|
        | 0   | PRB utilization ratio | 0                   | 1                 |
        | 1   | Cell throughput       | 0                   | 10*n_ues          |
        | 2   | UE SINRs              | 0.01 (-20 dB)       | 1000 (30 dB)      |


    Rewards:
    Similar to [1], it is possible to choose between two reward functions:
    
        1. Average network throughput 
        2. The inverse of the average deviation of the cell load

    By default, the average network throughput is used

    Starting State:
    At initialization, all the UEs are assigned to the eNodeB with the best 
    signal quality

    Version History:
    * v0: Initial versions release

    References:

    [1] Load Balancing in Cellular Networks: A Reinforcement Learning Approach
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9045699&tag=1

    Args:
        rew_type (int): Used to select the reward function type. (0: average throughput, 1: average deviation of the cell load). The default value is rew_type = 0 .
        hysteresis (int): The hysteresis value in dB. The default values is hysteresis = 3.
        include_Tput (bool): Whether to include the cell throughput in the observation space. The default value is include_Tput=False.
        include_sinr (bool): whether to include the UE SINR in the observation space. The default value is include_sinr=False.
    """

    def __init__(self,
                 *args,
                 rew_type=0,
                 hysteresis=3,
                 include_Tput=False,
                 include_sinr=False,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.cio_discrete_set = [-15, -6, -3, 0, 3, 6, 15]

        self.rew_type = rew_type
        self.hysteresis = hysteresis
        self.include_Tput = include_Tput
        self.include_sinr = include_sinr

        n_actions = len(self.cio_discrete_set)

        state_space_dim = self.n_cells
        state_high = 1.0
        state_low = 0.0

        if include_Tput:
            state_space_dim += self.n_cells
            state_high = 10*self.n_ues
            state_low = 0

        if include_sinr:
            state_space_dim += self.n_ues
            state_high = max(1000, state_high)
            state_low = min(0.01, state_low)

        self.state_space_dim = state_space_dim
        self.action_space = spaces.MultiDiscrete([n_actions]*self.n_cells)

        self.observation_space = spaces.Box(state_low,
                                            state_high,
                                            shape=(state_space_dim,))
    
    def reset(self):
        self._reset_sim()
        obs = self._sim.reset()
        out = self._process_sim_obs(obs)
        if self.log_movies:
            obs['reward'] = self._reward(obs)
            self._curr_run = [obs]
            self._total_steps = 0
            self._sum_reward = 0.
        self.last_obs = obs
        return out

    def _reset_sim(self):
        self._s_gen.set_random_state(self.random_state)
        self.scenario = self._s_gen.gen()
        self._sim = MultiBandCellularNetworkSimulator(self.scenario,
                                                      self.time_step_s)

    def _process_sim_obs(self, obs):
        cell_loads = obs['cells']['loads'].reshape(-1)
        state = cell_loads

        if self.include_Tput:
            cell_Tput = obs['cells']['Tput'].reshape(-1)
            state = np.concatenate([state, cell_Tput])

        if self.include_sinr:
            sinrs = obs['UEs']['sinr']
            state = np.concatenate([state, sinrs])

        assert state.shape == (self.state_space_dim,)

        return state

    def _reward(self, obs):
        if self.rew_type == 0:
            # reward is the mean throughput between cells
            meanTput = np.mean(obs['cells']['Tput'])

            return meanTput

        elif self.rew_type == 1:
            # std of the load
            loads = obs['cells']['loads']
            mean_load = np.mean(loads)
            std_load = np.sum(np.abs(loads-mean_load))

            return 1/(1+std_load)
        else:
            raise NotImplementedError(
                f"reward type unrecognized {self.rew_type}")

    def step(self, cio_indices):

        if cio_indices is None:
            # Baseline actions when no MLB is applied
            cios = np.zeros((self.n_cells,))
        else:
            cios = np.array(self.cio_discrete_set)[cio_indices]

        cios = np.tile(cios, (self.n_ues, 1)).T
        rsrp = self.last_obs['UEs']['rsrp'].reshape(cios.shape)

        serving_cells = self.last_obs['UEs']['serving_cell']
        assignment_matrix = np.eye(self.n_cells)[serving_cells]
        assignment_matrix = assignment_matrix.T

        # Only apply hysteresis to the serving cells
        new_rsrp = rsrp + cios + assignment_matrix*self.hysteresis

        assignments = np.argmax(new_rsrp, axis=0)

        obs = self._sim.step(list(assignments.astype(np.int32)))
        done = obs['done']
        self.last_obs = obs
        out_obs = self._process_sim_obs(obs)
        reward = self._reward(obs)

        if not self.log_movies:
            return out_obs, reward, done, self.last_obs
            
        # For visualization
        obs['reward'] = reward
        self._curr_run.append(obs)
        self._total_steps += 1
        self._sum_reward += reward
        if done:
            self._complete_run = self._curr_run
            avg_reward = self._sum_reward / self._total_steps
            if self._sum_reward > self._best_run_score:
                self._best_run = self._curr_run
                self._best_run_score = avg_reward
        return out_obs, reward, done, self.last_obs
