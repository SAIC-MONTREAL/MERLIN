import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BatchNormFeaturesExtractor(BaseFeaturesExtractor):
    '''
        input -> batch_norm -> feature_extractor -> batch_norm -> downstream
    '''
    def __init__(self, observation_space,
                 net_arch=[],
                 activation_fn=nn.ReLU,
                 normalize=True):

        features_dim = 0
        if len(net_arch) > 0:
            features_dim = net_arch[-1]
        else:
            features_dim = np.sum(observation_space.shape)

        super().__init__(observation_space, features_dim)
        self._normalize = normalize

        if len(net_arch) > 0:
            modules = [nn.Linear(np.sum(observation_space.shape), net_arch[0]), activation_fn()]
        else:
            modules = []

        for idx in range(len(net_arch) - 1):
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            modules.append(activation_fn())

        self._has_modules = len(modules) > 0
        if normalize:
            self.batch_norm = nn.BatchNorm1d(observation_space.shape)
        if self._has_modules:
            if normalize:
                self.nn_model = nn.Sequential(*modules, nn.BatchNorm1d(net_arch[-1]))
            else:
                self.nn_model = nn.Sequential(*modules)
        elif not normalize:
            self.nn_model = nn.Flatten()
            self._has_modules = True

    def batch_normalize(self, observations):
        return self.batch_norm(observations)

    def forward(self, observations):
        if self._normalize:
            observations = self.batch_norm(observations)
        if self._has_modules:
            return self.nn_model(observations)
        return observations
