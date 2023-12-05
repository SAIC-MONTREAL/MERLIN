"""
Non-learning baselines.
"""

import numpy as np
import pandas as pd
import torch

from saic5g.envs.per_ue_base import PerUeHandover
from saic5g.envs.diff_qsim_env import DiffQsimEnv
from saic5g.solvers.solvers import SolverBase
from saic5g.evaluators.evaluators import rollout
from saic5g.simulators.diffqsim import allocate_demand, packet_loss, tput_factor_map

obs_names = ['is_ss', 'rsrp', 'rsrq', 'received', 'dropped']


class PerUeBaselineBase(SolverBase):

    def load_params(self, path, checkpoint=None):
        return

    def pre_eval(self, env):
        assert isinstance(env, PerUeHandover)
        self.n_cell, self.n_ues, self.n_obs = env.observation_space.shape

    def cell_loads(self, obs):
        sat = np.sum(obs[:, :, 3], axis=1)
        unsat = np.sum(obs[:, :, 4], axis=1)

        return sat, unsat

    def to_dataframe(self, obs):
        cell_idx = np.tile(
            np.array([x for x in range(self.n_cell)]), self.n_ues)
        ue_idx = np.repeat(
            np.array([x for x in range(self.n_ues)]), self.n_cell)
        pd_idx = pd.MultiIndex.from_arrays(
            (cell_idx, ue_idx), names=['cell_id', 'ue_id'])
        out = pd.DataFrame(data=obs[cell_idx, ue_idx, :],
                           index=pd_idx, columns=obs_names)

        return out


class MaxRsrp(PerUeBaselineBase):
    name = 'max_rsrp'

    def get_action(self, obs):
        rsrp = obs[:, :, 1]
        max_rsrp = np.argmax(rsrp, axis=0)

        return max_rsrp


class MaxRsrq(PerUeBaselineBase):
    name = 'max_rsrq'

    def get_action(self, obs):
        rsrq = obs[:, :, 2]
        max_rsrq = np.argmax(rsrq, axis=0)

        return max_rsrq


class MinLoad(PerUeBaselineBase):
    """
    If a UE is experiencing any packet loss, hand over to the least loaded
    cell that has an RSRP that is either better than the serving RSRP or some threshold.
    """
    name = 'min_load'
    good_rsrp = -90

    def get_action(self, obs):
        df = self.to_dataframe(obs)
        df = df.reset_index(drop=False)
        ss = np.array(df[df['is_ss'] == 1].sort_values('ue_id')['cell_id'])

        dropping = df[(df['is_ss'] == 1) & (df['dropped'] > 0)]
        # If you are dropping and there exists a cell that is better in terms
        # of rsrp and load, then switch.

        # We will consider candidates where the RSRP is good or better than the current cell
        dropping['min_rsrp'] = np.minimum(dropping['rsrp'], MinLoad.good_rsrp)
        sat, unsat = self.cell_loads(obs)
        loads = sat + unsat
        dropping['max_load'] = loads[dropping['cell_id']]

        ho = df.merge(
            dropping[['ue_id', 'min_rsrp', 'max_load']], on=['ue_id'])
        ho['load'] = loads[ho['cell_id']]
        ho = ho[(ho['rsrp'] >= ho['min_rsrp']) &
                (ho['load'] <= ho['max_load'])]
        ho = ho.sort_values('load').drop_duplicates(
            subset='ue_id', keep='first')
        ss[np.array(ho['ue_id'])] = np.array(ho['cell_id'])

        return ss


class A3(PerUeBaselineBase):
    """
    A3 algorithm with hyteresis term.
    """
    name = 'a3'

    def get_action(self, obs):
        hys = 2.
        ss = np.argmax(obs[:, :, 0], axis=0)
        rsrp = obs[:, :, 1]
        n_cell, n_ue = rsrp.shape

        for ue_id in range(n_ue):
            ue_rsrp = rsrp[:, ue_id]
            s = ss[ue_id]
            s_rsrp = ue_rsrp[s]
            t = np.argmax(ue_rsrp)
            t_rsrp = ue_rsrp[t]

            if s_rsrp < t_rsrp-hys:
                ss[ue_id] = t

        return ss


class StaticOptimal(PerUeBaselineBase):
    """
    TODO: if we release this one, we should also put the paper on Arxiv and
    link to it.
    """
    name = 'static_optimal'

    def pre_eval(self, env):
        """
        Roll out the env using maxrsrp heuristic.
        """
        super().pre_eval(env)
        max_rsrp = MaxRsrp(None)
        max_rsrp.pre_eval(env)
        rollout(env, max_rsrp)
        history = env.get_last_complete_run()
        cell_capacity = env.scenario.radio.cell_capacity(0, env.n_cells)
        opt = StaticAssignmentOptimizer()
        self.actions = opt.throughput_optimal_actions(
            history, cell_capacity, lookahead=True)
        self.curr_step = 0

    def get_action(self, obs):
        out = self.actions[self.curr_step]
        self.curr_step += 1

        return out


class GradOpt(PerUeBaselineBase):
    """
    Do gradient descent using differentiable QSim to optimize assignment.

    Note that this is very slow on CPU.
    """
    name = 'grad_opt'

    def pre_eval(self, env):
        super().pre_eval(env)
        assert isinstance(env, DiffQsimEnv)
        # needed to initialize scenario
        env.reset()
        self.n_iters = 100
        self.grid_shape = (100, 100)
        self.cell_pos = env.scenario.geom.cell_pos()
        self.n_cells = len(self.cell_pos)
        self.cell_capacities = env.scenario.radio.cell_capacity(
            0, self.n_cells)
        self.bbox = env.scenario.geom.bbox
        ue_height = env.scenario.geom.ue_pos(0)[0, 2]
        self.tput_factor_map = tput_factor_map(
            env.scenario.radio,
            self.cell_pos,
            self.bbox,
            self.grid_shape,
            ue_height=ue_height)

        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

    def parse_obs(self, obs):
        return obs[:, :2], obs[:, 2]

    def get_action(self, obs):
        # obs should contain ue positions and demands
        # initialize assignments based on best tput factor
        ue_pos, ue_demand = self.parse_obs(obs)
        grid_pos_x = (ue_pos[:, 0] - self.bbox[0]) / \
            (self.bbox[2] - self.bbox[0])
        grid_pos_y = (ue_pos[:, 1] - self.bbox[1]) / \
            (self.bbox[3] - self.bbox[1])

        mean = torch.zeros((self.n_cells, len(ue_pos)))
        std = torch.ones((self.n_cell, len(ue_pos)))
        assigns = torch.normal(mean, std).to(self.device)
        assigns.requires_grad = True
        ue_pos = torch.Tensor(
            np.stack([grid_pos_x, grid_pos_y], axis=1)).to(self.device)
        ue_demand = torch.Tensor(ue_demand).to(self.device)
        tput_factors = torch.Tensor(self.tput_factor_map).to(self.device)
        cell_capacities = torch.Tensor(self.cell_capacities).to(self.device)
        opt = torch.optim.SGD([assigns], lr=.01, momentum=0.9)
        best_pl = 1

        for i in range(self.n_iters):
            ues = allocate_demand(
                ue_pos,
                ue_demand,
                assigns,
                self.grid_shape,
            )
            loss = packet_loss(ues, tput_factors, cell_capacities)

            if i % 10 == 0:
                print(i, loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ld = loss.detach()

            if ld < best_pl:
                best_pl = ld
                best_assign = assigns.detach().cpu().numpy().argmax(axis=0)

        return best_assign


class NoLb(PerUeBaselineBase):
    name = "no_lb"

    def pre_eval(self, env):
        pass

    def get_action(self, obs):
        return None
