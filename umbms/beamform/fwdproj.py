"""
Tyson Reimer
University of Manitoba
June 3rd, 2019
"""

import numpy as np
import multiprocessing as mp
from functools import partial

###############################################################################


def fd_fwd_proj(model, phase_fac, dv, freqs, n_cores=2):
    """Forward project in the frequency domain

    Parameters
    ----------
    model : array_like, NxN
        Image-space model to be forward projected. N is number of pixels
        along one dimension.
    phase_fac : array_like, MxNxN
        Phase factor for efficient computation. N is number of pixels
        along one dimension of model, M is number of antenna positions
        used in the scan.
    dv : float
        Volume element, units m^3 (also area element, units m^2)
    freqs : array_like, Fx1
        The frequency vector used in the scan, in Hz, for F frequencies
    n_cores : int
        Number of cores to use for parallel processing

    Returns
    -------
    fwd : array_like, LxM
        Forward projection of primary scatter responses only. L is the
        number of frequencies in the scan, M is the number of antenna
        positions.
    """

    n_fs = np.size(freqs)

    # Create function for parallel processing
    parallel_func = partial(_parallel_fd_fwd_proj, freqs,
                            phase_fac, model, dv)

    workers = mp.Pool(n_cores)  # Init worker pool

    iterable_idxs = range(n_fs)  # Get indices to iterate over

    # Collect forward projections
    fwds = np.array(workers.map(parallel_func, iterable_idxs))

    fwds = np.reshape(fwds, [n_fs, 72])
    workers.close()

    return fwds


def _parallel_fd_fwd_proj(freqs, phase_fac, model, dv, ff):
    """Parallel processing function to compute projection at freq ff

    Parameters
    ----------
    freqs : array_like, Lx1
        Frequency vector, L is the number of frequencies
    phase_fac : array_like, MxNxN
        Phase factor for efficient computation. N is number of pixels
        along one dimension of model, M is number of antenna positions
        used in the scan.
    model : array_like, NxN
        Image-space model to be forward projected. N is number of pixels
        along one dimension.
    ff : int
        Frequency index

    Returns
    -------
    p_resp : array_like, LxM
        Forward projection at frequency ff
    """

    temp_var = phase_fac**freqs[ff]

    temp_var2 = model[None, :, :] * temp_var

    # Compute the primary scatter responses
    p_resp = np.sum(temp_var2 * temp_var, axis=(1, 2)) * dv

    return p_resp
