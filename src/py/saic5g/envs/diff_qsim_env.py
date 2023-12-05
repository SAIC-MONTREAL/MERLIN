from gym import spaces
import numpy as np

from saic5g.envs.per_ue_base import PerUeHandover


class DiffQsimEnv(PerUeHandover):
    """
    Environment designed to be used with differentiable QSim.

    Returned observations describe UE positions and demands.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _observe_sim_info(self):
        o = self.last_obs
        return np.concatenate([o['UEs']['position'][:,:2], o['UEs']['demand'][:, None]], axis=1)

    def reset(self):
        super().reset()
        return self._observe_sim_info()

    def step(self, assignments):
        _, reward, done, info = super().step(assignments)
        return self._observe_sim_info(), reward, done, info