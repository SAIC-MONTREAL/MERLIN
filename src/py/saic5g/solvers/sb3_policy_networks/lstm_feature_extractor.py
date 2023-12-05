import torch as th
import torch.nn as nn
from torch.autograd import Variable 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomLSTM(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim=64,
                 num_layers=2, bidirectional=1,
                 activation=nn.ReLU):
        self._num_layers = num_layers
        self._bidirectional = bidirectional
        self._n = self._num_layers
        self._hidden_size = features_dim
        if self._bidirectional:
            self._n *= 2
            features_dim *= 2

        super(CustomLSTM, self).__init__(observation_space, features_dim)

        self.lstm = nn.LSTM(input_size=observation_space.shape[-1], hidden_size=self._hidden_size,
                            num_layers=self._num_layers, batch_first=True,
                            bidirectional=bool(self._bidirectional))

        self.activation = activation()

    def forward(self, x):
        h_0 = Variable(th.zeros(self._n, x.size(0), self._hidden_size)) #hidden state
        c_0 = Variable(th.zeros(self._n, x.size(0), self._hidden_size)) #internal state

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        if self._bidirectional:
            output = th.cat((output[:, -1, :self._hidden_size], output[:, 0, self._hidden_size:]), dim=-1)
        else:
            output = output[:, -1]
        out = self.activation(output)

        return out
