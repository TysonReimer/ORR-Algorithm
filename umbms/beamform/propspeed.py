"""
Tyson Reimer
University of Manitoba
November 8 2018
"""

import numpy as np

from umbms.beamform.breastmodels import get_breast, get_roi
from umbms.beamform.extras import apply_ant_t_delay

###############################################################################

__VAC_SPEED = 3e8  # Define propagation speed in vacuum

###############################################################################


def estimate_speed(adi_rad, ant_rad, m_size=500, new_ant=True):
    """Estimates the propagation speed of the signal in the scan

    Estimates the propagation speed of the microwave signal for
    *all* antenna positions in the scan. Estimates using the average
    propagation speed.

    Parameters
    ----------
    adi_rad : float
        The approximate radius of the breast, in m
    ant_rad : float
        The radius of the antenna trajectory in the scan, as measured
        from the black line on the antenna holder, in m
    m_size : int
        The number of pixels along one dimension used to model the 2D
        imaging chamber
    new_ant : bool
        If True, indicates the 'new' antenna (from 2021) was used
    Returns
    -------
    speed : float
        The estimated propagation speed of the signal at all antenna
        positions, in m/s
    """

    # Correct for antenna phase-delay
    ant_rad = apply_ant_t_delay(ant_rad, new_ant=new_ant)

    # Model the breast as a homogeneous adipose circle, in air
    breast_model = get_breast(m_size=m_size, adi_rad=adi_rad, ant_rad=ant_rad)

    # Get the region of the scan area within the antenna trajectory
    roi = get_roi(ant_rad, m_size, ant_rad)

    # Estimate the speed
    speed = np.mean(__VAC_SPEED / np.sqrt(breast_model[roi]))

    return speed
