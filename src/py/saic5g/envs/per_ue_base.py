from collections import defaultdict

from gym import Env, spaces
from gym.utils import seeding
import numpy as np
import pandas as pd

from saic5g.simulators.qsim import QSim
from saic5g.vis.sim import SimVis


class PerUeHandover(Env):
    """
    Class which supports fine-grained control of UE-cell assignments.
    
    UE-level control often leads to very large action spaces which are difficult
    to tackle with most RL methods. However, this class is intended to be used as 
    a base for others with more manageable action spaces, such as the the Cio env.
    The observation space of this class is also very large, outputting information
    needed to run different baselines, but not realistically useable for RL. Derived 
    classes should access self.last_obs to access simulator outputs and parse those 
    to create observations.

    This env assumes that the number of UEs and cells in the simulation do not
    change with time.

    Action space: UE-cell assignments
    Observation space: n_cells x n_ues x 5. For each cell-ue pair, the observation includes
        (in order along the last dimension)
        - is cell serving UE (0 if no, 1 if yes)?
        - rsrp
        - rsrq
        - amount of UE demand successfully served through the cell (0 if not the serving cell)
        - amount of UE demand dropped through the cell (0 if not the serving cell)
    Reward: Total data volume successfully transmitted by the system
    Infos: Per-cell and Per-UE transmitted data volume
    """

    def __init__(self,
                 scenario_generator,
                 time_step_s=1.,
                 log_movies=True):
        """
        Args:
            scenario_generator(ScenarioGenerator): Generator for scenarios.
            time_step_s (float): simulation time step
            log_movies (bool): whether to store state information to facilitate movie creation.
        """
        self.time_step_s = time_step_s
        self._s_gen = scenario_generator
        self._one_obs = 5
        self._complete_run = []
        self._reset_best()
        self.seed()
        self.log_movies = log_movies
        self._s_gen.set_random_state(self.random_state)
        s = self._s_gen.gen()
        self.n_ues = len(s.geom.ue_pos(0))
        self.n_cells = len(s.geom.cell_pos())

        self.action_space = spaces.Box(0, self.n_cells - 1, shape=(self.n_ues,), dtype=np.int32)
        # -inf to inf not great for scaling observations, but OK since this class unlikely to be used for RL.
        self.observation_space = spaces.Box(np.NINF, np.Inf, shape=(self.n_cells, self.n_ues, self._one_obs))

    def _reset_sim(self):
        self._s_gen.set_random_state(self.random_state)
        self.scenario = self._s_gen.gen()
        self._sim = QSim(self.scenario, self.time_step_s)

    def seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        self._s_gen.set_random_state(self.random_state)
        return [seed]

    def _process_sim_obs(self, obs):
        obs = obs['UEs']
        out_obs = np.zeros((self.n_cells, self.n_ues, self._one_obs))
        ue_idx = np.arange(self.n_ues)
        serving_cell_idx = obs['serving_cell'][ue_idx]
        out_obs[serving_cell_idx, ue_idx, 0] = 1.
        out_obs[:, :, 1] = obs['rsrp']
        out_obs[:, :, 2] = obs['rsrq']
        demand = obs.get('demand', 1.)
        out_obs[serving_cell_idx, ue_idx, 3] = obs['packet_loss'] * demand
        out_obs[serving_cell_idx, ue_idx, 4] = (1. - obs['packet_loss']) * demand
        return out_obs

    def reset(self):
        self._reset_sim()
        obs = self._sim.reset()
        out = self._process_sim_obs(obs)
        if self.log_movies:
            ue_reward, cell_reward = self._rewards(obs)
            obs['reward'] = ue_reward.sum()
            self._curr_run = [obs]
            self._total_steps = 0
            self._sum_reward = 0.
        self.last_obs = obs
        return out

    def _rewards(self, obs):
        per_ue_reward = (1 - obs['UEs']['packet_loss'])*obs['UEs']['demand']
        per_cell_reward = (1 - obs['cells']['packet_loss'])*obs['cells']['demand']
        return per_ue_reward, per_cell_reward

    def step(self, assignments):
        obs = self._sim.step(list(assignments.astype(np.int32)))
        done = obs['done']
        self.last_obs = obs
        out_obs = self._process_sim_obs(obs)
        ue_rewards, cell_rewards = self._rewards(obs)
        reward = ue_rewards.sum()

        info = {
            'cell_rewards': cell_rewards,
            'ue_rewards': ue_rewards,
            'simulation_time': obs['simulation_time']
        }

        if not self.log_movies:
            return out_obs, reward, done, info

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
        return out_obs, reward, done, info

    def make_movie(self, path, which='best'):
        """
        Visualize either the best or last complete simulation run 
        as a video and write it to path.

        Args:
            path (Path or str): path to write the result
            which (str): 'best' or 'last'
        """
        self._check_can_make_movie()
        if which == 'best':
            run = self._best_run
        elif which == 'last':
            run = self._complete_run
        else:
            raise ValueError()
        bounds = SimVis.bounds(run)
        vis = SimVis(self.n_cells, self.n_ues, bounds)
        vis.create_movie(run, path, 800, vis.height_px(800))

    def get_last_complete_run(self):
        """
        Get the list of simulator outputs from the last complete rollout.

        Returns:
            list
        """
        self._check_can_make_movie()
        return self._complete_run

    def _check_can_make_movie(self):
        if not self.log_movies:
            raise RuntimeError('Requested a visualization, but movie logging is turned off.')
        if len(self._complete_run) < 2:
            print("Warning: requested to visualize last complete simulation, but none exists.")
            return

    def _reset_best(self):
        self._best_run = []
        self._best_run_score = float("-inf")