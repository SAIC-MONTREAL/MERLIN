import numpy as np

from saic5g.scenarios.interfaces import Generator, Scenario


class BasicScenario(Scenario):
    def initial_assignment(self):
        dist, az = self.geom.cell_ue_distance(0), self.geom.cell_ue_azimuth(0)
        rsrp = self.radio.rsrp(0, dist, az)
        return np.argmax(rsrp, axis=0)

class ScenarioGenerator(Generator):
    """
    Container to hold all generators.
    """
    def __init__(self, 
                 geom_generator, 
                 traffic_generator, 
                 radio_generator,
                 scen_class=BasicScenario):
        """
        Args:
            geom_generator (Generator)
            traffic_generator (Generator)
            radio_generator (Generator)
            scen_class (Class)
        """
        self.gens = (geom_generator, traffic_generator, radio_generator)
        self.scen_class = scen_class

    def set_random_state(self, random_state):
        for g in self.gens:
            g.set_random_state(random_state)

    def gen(self):
        return self.scen_class(*[g.gen() for g in self.gens])


class BasicSubconfigGenerator(Generator):
    """
    A subconfig generator which returns the same subconfig every time.

    Useful for subconfigs you don't need to modify between env executions.
    """
    def __init__(self, subconfig):
        """
        Args:
            subconfig (ScenarioSubconfig)
        """
        self._subconfig = subconfig
        self.rs = None

    def gen(self):
        if self.rs is None:
            raise RuntimeError('Please set random state.')
        self._subconfig.set_random_state(self.rs)
        return self._subconfig