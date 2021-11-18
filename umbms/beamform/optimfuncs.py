"""
Tyson Reimer
University of Manitoba

"""

import numpy as np
import multiprocessing as mp
from functools import partial

###############################################################################


def get_ref_derivs(phase_fac, fd, fwd, freqs, n_cores=2):
    """Get the gradient of the loss func wrt the reflectivities

    Parameters
    ----------
    phase_fac : array_like
        The phase factor of the imaging domain
    fd : array_like
        The frequency domain measured data
    fwd : array_like
        The frequency domain forward projection of the current image
        estimate
    freqs : array_like
        The frequencies used in the scan
    n_cores : int
        The number of cores to use during the parallelization

    Returns
    -------
    ref_derives : array_like
        The gradient of the loss function with respect to the
        reflectivities in the image domain
    """

    # Create func for parallel computation
    parallel_func = partial(_parallel_ref_deriv, phase_fac, fwd, fd,
                            freqs)

    workers = mp.Pool(n_cores)  # Init worker pool

    iterable_idxs = range(np.size(fd, axis=0))  # Indices to iterate over

    # Store projections from parallel processing
    all_ref_derivs = np.array(workers.map(parallel_func, iterable_idxs))

    # Reshape
    all_ref_derivs = np.reshape(all_ref_derivs, [np.size(fd, axis=0),
                                                 np.size(phase_fac, axis=1),
                                                 np.size(phase_fac, axis=2)])

    workers.close()  # Close worker pool

    # Sum over all frequencies
    ref_derivs = np.sum(all_ref_derivs, axis=0)

    ref_derivs *= -1  # Apply normalization factor

    return ref_derivs


def _parallel_ref_deriv(phase_fac, fwd, fd, freqs, ff):
    """Parallelized function for ref_deriv calculation

    Parameters
    ----------
    phase_fac : array_like
        The phase factor of the imaging domain
    fd : array_like
        The frequency domain measured data
    fwd : array_like
        The frequency domain forward projection of the current image
        estimate
    freqs : array_like
        The frequencies used in the scan
    ff : int
        The index for the frequency to be used

    Returns
    -------
    ref_deriv : array_like
        The gradient at this frequency
    """

    # Calculate the derivative wrt to the S-parameter
    s_deriv = phase_fac**(2 * freqs[ff])

    # Calculate the derivative wrt the reflectivities
    ref_deriv = np.sum((np.conj(s_deriv)
                        * (fd[ff, :, None, None] - fwd[ff, :, None, None])
                        + s_deriv * np.conj(fd[ff, :, None, None]
                                            - fwd[ff, :, None, None])),
                       axis=0)

    return ref_deriv
