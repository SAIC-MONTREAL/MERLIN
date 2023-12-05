"""
Scenario configuration interfaces.

The aims of this interface are to a) organize the configuration into the (somewhat) orthogonal
notions of geometric, radio, and traffic (how much demand each UE creates) configuration. In addition,
it aims to allow the user to control how scenarios are randomized every time the simulation is reset.
To this end, we define specific Geom, Radio, and Traffic interfaces which define all of the parameters
for a single simulation. We also define a Scenario interface to group all of these three to make them 
easier to pass around.
"""


class ScenarioSubconfig:
    """
    Base class for all scenario subconfigurations (Geom, Traffic, Radio)
    """

    def set_random_state(self, random_state):
        self.rs = random_state


class Scenario:
    """
    Container for holding subconfigs.

    Also defines assingment initialization behavior.
    """

    def __init__(self, geom, traffic, radio):
        """
        Args:
            geom (ScenarioSubconfig)
            traffic (ScenarioSubconfig)
            radio (ScenarioSubconfig)
        """
        self.geom = geom
        self.traffic = traffic
        self.radio = radio

    def initial_assignment(self):
        """
        Get initial assignment of UEs to cells.

        Returns:
            np.Array: (n_ues,)
        """
        raise ValueError


class Generator:
    """
    Base class for generators of Scenario or ScenarioSubconfig
    """

    def set_random_state(self, random_state):
        self.rs = random_state

    def gen(self):
        """
        Return a new Scenario / ScenarioSubconfig

        Returns:
            Scenario or ScenarioSubconfig
        """
        raise NotImplementedError



class Geom(ScenarioSubconfig):
    """
    Base class for geometry-related configuration.

    This class is also responsible for controlling when the simulation ends.
    """

    def cell_pos(self):
        """
        Get cell positions

        Returns:
            np.array (n_cell, 3): cell XYZ locations in meters
        """
        raise NotImplementedError

    def ue_pos(self, t):
        """
        Get UE positions.

        Args:
            t (float): time in seconds

        Returns:
            np.array (n_ues, 3): UE xyz positions in meters
        """
        raise NotImplementedError

    def cell_ue_distance(self, t):
        """
        Get matrix representing pairwise distance between cells and UEs

        Args:
            t (float): sim time

        Returns:
            np.array (n_cell, n_ue)
        """
        raise NotImplementedError

    def cell_ue_azimuth(self, t):
        """
        Get matrix representing pairwise azimuth between cells and UEs

        Args:
            t (float): sim time

        Returns:
            np.array (n_cell, n_ue)
        """
        raise NotImplementedError

    def end_time(self):
        """
        Get end time.

        Returns:
            float: end time in seconds
        """
        raise NotImplementedError

    def done(self, t):
        """
        Check if time is greater than end time

        Args:
            t(float): time in seconds

        Returns:
            bool
        """
        raise NotImplementedError



class Radio(ScenarioSubconfig):
    """
    Base class for radio - related parameters for the QSim simulator.

    Numerical values should NOT be in decibels.
    """

    def handover_drop_load(self, t, assigns, last_assigns):
        """
        Compute the load dropped due to handovers.

        Args:
            t (float): simulation time
            assigns (n_ues) np.array: current UE/cell assignments
            last_assigns (n_ues) np.array: previous UE/cell assignments

        Returns:
            (n_ues) np.array: per-ue packet loss associated with handover.
                Values should range between 0 and 1.
        """
        raise NotImplementedError()

    def n_prb(self, t, n_cell):
        """
        Number of physical resource blocks available to cells.

        Args:
            t (float): simulation time
            n_cell (int): number of cells

        Returns:
            (n_cell) np.array: per-cell numbers of resource blocks
        """
        raise NotImplementedError()

    def interference(self, t, n_cell, n_ues):
        """
        Get inteference in the connection between cells UEs.

        Args:
            t (float): simulation time
            n_cell (int): number of cells
            n_ues (int): number of UEs

        Returns:
            (n_cell, n_ues) np.array: Interference between each cell and each UE.
        """
        raise NotImplementedError()

    def noise(self, t, n_cell, n_ues):
        """
        Get noise in the connection between cells UEs.

        Args:
            t (float): simulation time
            n_cell (int): number of cells
            n_ues (int): number of UEs

        Returns:
            (n_cell, n_ues) np.array: Noise between each cell and each UE.
        """
        raise NotImplementedError()

    def activity_factor(self, t, n_cell):
        """
        cell Activity factor.

        See https://www.cablefree.net/wirelesstechnology/4glte/lte-rsrq-sinr/ for more
        details on this quantity.

        Args:
            t (float): simulation time
            n_cell (int): number of cells

        Returns:
            (n_cell) np.array: per-cell activity factors.
        """
        raise NotImplementedError()

    def cell_capacity(self, t, n_cell):
        """
        Return the max capacity of each cell.

        args:
            t (float): simulation time
            n_cell (int): number of cells

        returns:
            (n_cell) np.array: per-cell maximum capacities.
        """
        raise NotImplementedError()

    def rsrp(self, t, distance, azimuth):
        """
        Compute RSRP as a function of distance from a given cell.

        args:
            t (float): simulation time
            distance (n_cell, n_ues) np.array: distances (in meters)
                between each cell and each UE
            azimuth (n_cell, n_ues) np.array: angle (in radians) between
                each cell and each UE wrt (1, 0) vector.

        returns:
            (n_cell, n_ues) np.array: RSRP values between each cell and each UE.
        """
        raise NotImplementedError()


class Traffic(ScenarioSubconfig):
    """
    Base class for traffic - related parameters for the QSim simulator.

    Numerical values should NOT be in decibels.
    """

    def ue_demand(self, t, n_ues):
        """
        Compute the "demand" of a UE at a given time.

        QSim does not make a distinction between uplink and downlink,
        so this is a single value for each UE.

        args:
            t (float): simulation time
            n_cell (int): number of cells

        returns:
            (n_ues) np.array: per-ue demand. Cannot be negative.
        """
        raise NotImplementedError()