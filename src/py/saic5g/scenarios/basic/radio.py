"""
Basic radio-related scenario configuration.

The functionality in this module encodes the physics of the radio communications. This includes
cell directionality, band, bandwidth, noise, path loss, etc.

TODO: Ask Amal if she can clean up these transmission models
"""
import math

import numpy as np

from saic5g.scenarios.interfaces import Radio, Generator
from saic5g.utils.radio import from_db
from saic5g.utils.np_utils import pi_wrap


class CellIndependentRadio(Radio):
    """
    A radio configuration where all the cells are the same (and omnidirectional).
    """
    # parameters for sampling interference and noise
    # NOTE: these were tuned (along with the activity factor) to try to achieve the following state:
    # rsrp = -90, rsrq = -15, sinr = 13
    # These numbers are apparently reasonable according to the table in
    # https://www.cablefree.net/wirelesstechnology/4glte/lte-rsrq-sinr/
    #  _in_mag = 6. * 10**(-9)
    _in_mag = 6. * 10**(-8)
    i_mean = 1. * _in_mag
    i_std = 0.1 * _in_mag
    n_mean = 1. * _in_mag
    n_std = 0.1 * _in_mag

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def handover_drop_load(self, t, assigns, last_assigns):
        drop = self.kwargs.get('handover_drop_load', 0.15)

        return (assigns != last_assigns).astype(np.float) * drop

    def n_prb(self, t, n_cell):
        return np.ones(n_cell) * self.kwargs.get('n_prb', 6)

    def interference(self, t, n_cell, n_ues):
        m = self.kwargs.get('i_mean', self.i_mean)
        s = self.kwargs.get('i_std', self.i_std)

        return self.rs.normal(m, s, (n_cell, n_ues))

    def noise(self, t, n_cell, n_ues):
        m = self.kwargs.get('n_mean', self.n_mean)
        s = self.kwargs.get('n_std', self.n_std)

        return self.rs.normal(m, s, (n_cell, n_ues))

    def sui_suburban_path_loss(self, d):
        """
        https://www.xirio-online.com/help/en/sui.html
        https://www1.cse.wustl.edu/~jain/wimax/ftp/channel_model_tutorial.pdf
        """
        f_ghz = 2.3
        c_gmps = 0.3  # speed of light in giga-meters / s
        lam = c_gmps/f_ghz  # wavelength in meters
        d0 = 100
        A = 20 * np.log10(4 * np.pi * d0 / lam)
        # using category B here
        # 32 is bs height in m
        gamma = 4 - 0.0065*32 + 17.1/32
        shadowing = 9  # usually lognormal 8.2 to 10.6 db

        return A + 10*gamma*np.log10(d/d0) + shadowing

    def rsrp(self, t, distance, azimuth):
        #  antenna_power = self.kwargs.get('dl_tx_power', 43.)
        antenna_power = self.kwargs.get('dl_tx_power', 40.)  # 10 w antenna
        #  path_loss = 128.1 + 37.6*np.log10(distance / 1000.) + 20*math.log10(2.3/2)
        sui_loss = self.sui_suburban_path_loss(distance)
        #  return from_db(antenna_power - path_loss)

        return from_db(antenna_power - sui_loss)

    def activity_factor(self, t, n_cell):
        # This is tuned, along with noise, to achieve a -15db rsrq at -90db rsrp.
        # A value higher than 1 indicates multiple tx antennas, but I think that
        # is not unreasonable for downlink.

        return np.ones(n_cell) * self.kwargs.get('activity_factor', 3.)

    def cell_capacity(self, t, n_cell):
        return np.ones(n_cell) * self.kwargs.get('cell_capacity', 10.)


class SectorAntenna120:
    """
    Antenna model comes from me eyeballing this datasheet for an LTE 120 degree sector antenna
    https://www.kpperformance.com/Content/Images/Downloadables/Datasheets/KP-2DP120S-45_datasheets_US.pdf

    I couldn't find numerical data in non-graphical form after ~30 minutes searching.
    """

    def __init__(self):
        # -3DB point (half power) at 60 degrees, making this a 120 degree antenna
        self.angle = np.radians(np.array([0, 60, 90, 120, 180]))
        self.gain = np.array([0, -3, -10, -20, -100])

    def dir_gain(self, angle):
        """
        Angles in radians
        """
        angle = np.abs(pi_wrap(angle))

        return np.interp(angle, self.angle, self.gain)


class OmniDirectionalAntenna:

    def dir_gain(self, angle):
        return 0


class SectorRadio(CellIndependentRadio):
    """
    A radio configuration where you can have either 1 or 3 sectors per base station,
    and 3 or 4 bands per sector. This radio should be paired with a geometry where there
    are n_sectors * n_bands cells in a row in the same place, followed by the next
    n_secotrs * n_bands cells in a row in the next place, and so on.
    """

    def __init__(self, sectors='DEFAULT_1_SECTOR', bands='DEFAULT_3_BAND', **kwargs):
        super().__init__(**kwargs)

        if sectors == 'DEFAULT_3_SECTOR':
            self.n_sectors = 3
            self.antenna = SectorAntenna120()
        elif sectors == 'DEFAULT_1_SECTOR':
            self.n_sectors = 1
            self.antenna = OmniDirectionalAntenna()
        else:
            raise ValueError(f'Unknown "sectors" value {sectors}')

        step = 2*np.pi / self.n_sectors
        self.sector_azimuth = np.arange(self.n_sectors) * step

        if bands == 'DEFAULT_3_BAND':
            self.n_freq = 3
            self._frequency_loss_map = np.array([
                20*math.log10(2.3/2),
                20*math.log10(1.8/2),
                20*math.log10(0.850/2)
            ])
            self._capacity_map = np.array([
                160.,
                40,
                40.
            ])
        elif bands == 'DEFAULT_4_BAND':
            self.n_freq = 4
            self._frequency_loss_map = np.array([
                20*math.log10(2.3/2),
                20*math.log10(2.3/2),
                20*math.log10(1.8/2),
                20*math.log10(0.850/2)
            ])
            self._capacity_map = np.array([
                100.,
                50.,
                25,
                50.
            ]) / 10  # TODO: don't forget to remove
        else:
            raise ValueError(f'Unknown "bands" value {bands}')

    def _band_idx(self, cell_idx):
        return cell_idx % self.n_freq

    def _sector_idx(self, cell_idx):
        return (cell_idx // self.n_freq) % self.n_sectors

    def cell_capacity(self, t, n_cell):
        return self._capacity_map[self._band_idx(np.arange(n_cell))]

    def rsrp(self, t, distance, azimuth):
        n_cell = distance.shape[0]
        cell_idx = np.arange(n_cell)
        # Input azimuth is expressed wrt to (1, 0) vector, but we need to rotate wrt to the
        # antenna pointing vector for each cell.
        antenna_azimuth = np.repeat(self.sector_azimuth, self.n_freq)
        antenna_azimuth = np.tile(
            antenna_azimuth, n_cell // (self.n_freq * self.n_sectors))
        azimuth = antenna_azimuth[:, None] - azimuth
        # Default is small-cell
        antenna_power = self.kwargs.get(
            'dl_tx_power', 30.) + self.antenna.dir_gain(azimuth)
        path_loss = 128.1 + 37.6 * \
            np.log10(distance / 1000.) + \
            self._frequency_loss_map[cell_idx % self.n_freq, None]

        return from_db(antenna_power - path_loss)


class MultiBandCellularRadio(Radio):
    """
    A simulation of macro-cellular network with one sector antennas.
    Each base station can operate one or multiple bands
    """

    def __init__(self, **kwargs):

        self.n_sectors = 1
        self.antenna = OmniDirectionalAntenna()

        self.frequencies = kwargs.get("frequencies")  # in MHz

        if self.frequencies is None:
            raise ValueError(f"Frequencies cannot be None")

        self.n_freq = len(self.frequencies)

        self.tx_powers = kwargs.get("tx_powers")  # in dBm

        if self.tx_powers is None:
            raise ValueError("Transmit powers cannot be None")

        assert len(self.frequencies) == len(self.tx_powers)

        self.resource_blocks = kwargs.get("resource_blocks")
        if self.resource_blocks is None:
            raise ValueError(
                "Number of resource block per band cannot be None")
        self.resource_blocks = np.array(self.resource_blocks)

        self.sub_carriers_per_rb = 12

    def _band_idx(self, cell_idx):
        return cell_idx % self.n_freq

    def _sector_idx(self, cell_idx):
        return (cell_idx // self.n_freq) % self.n_sectors

    def cell_capacity(self, t, n_cell):
        return self.resource_blocks[self._band_idx(np.arange(n_cell))]

    def compute_path_loss(self, carrier_index, distances):
        """
        Taken from 3GPP R 25.814 Table A.2.1.1-3
        """

        central_freq_mhz = self.frequencies[carrier_index]

        log10_dist_km = np.log10(distances/1000)

        if (central_freq_mhz >= 700 and central_freq_mhz < 1000):
            return 120.9 + 37.6*log10_dist_km + 20 * np.log10(central_freq_mhz / 900)
        elif (central_freq_mhz >= 1300 and central_freq_mhz < 2700):
            return 128.1 + 37.6*log10_dist_km + 20 * np.log10(central_freq_mhz / 2000)
        else:
            raise ValueError(f"Path loss not defined for {central_freq_mhz}")

    def rsrp(self, t, distances, azimuth):
        """
        Compute UE received power and RSRP 
        Received power (UE) = Transmit power (BS) + BS Antenna gain \
                + UE Antenna gain – Pathloss – Penetration loss + Shadowing
        RSRP = Total received power / # of subcarriers
        Assume RS power = Data power
        """
        n_cells, n_ues = distances.shape[0], distances.shape[1]

        n_sites = int(n_cells/self.n_freq)

        distances = distances.reshape((n_sites, self.n_freq, n_ues))  # in m

        total_rxp_dbm = np.zeros_like(distances)
        rsrp_dbm = np.zeros_like(distances)

        for carrier_index in range(self.n_freq):
            path_loss = self.compute_path_loss(carrier_index,
                                               distances[:, carrier_index, :])

            received_power = self.tx_powers[carrier_index] - path_loss

            total_rxp_dbm[:, carrier_index, :] = received_power

            totalNumberOfSubcarriers = self.resource_blocks[carrier_index] * \
                self.sub_carriers_per_rb

            subCarrier_log10 = 10 * np.log10(totalNumberOfSubcarriers)

            rsrp = received_power - subCarrier_log10
            rsrp_dbm[:, carrier_index, :] = rsrp

        return total_rxp_dbm, rsrp_dbm
