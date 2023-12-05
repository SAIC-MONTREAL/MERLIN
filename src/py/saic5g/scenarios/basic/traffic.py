"""
Basic traffic-related scenario configuration.

TODO: Ask Amal to add a poisson process or something
"""
import numpy as np
from saic5g.scenarios.interfaces import Traffic


class ConstantTraffic(Traffic):
    """
    A traffic configuration that generates constant traffic for each UE.
    Assumes data is in kb
    """

    def __init__(self, data_size_kb=12):
        self.data_size_kb = data_size_kb

    def ue_demand(self, t, n_ues):
        return np.ones(n_ues)*self.data_size_kb
