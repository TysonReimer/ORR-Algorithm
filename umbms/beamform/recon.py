"""
Tyson Reimer
University of Manitoba
June 4th, 2019
"""

import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

from functools import partial

from umbms import null_logger

from umbms.loadsave import save_pickle

from umbms.beamform.extras import get_pix_ts, get_fd_phase_factor

from umbms.beamform.fwdproj import fd_fwd_proj
from umbms.beamform.optimfuncs import get_ref_derivs

from umbms.plot import plt_sino, plt_fd_sino
from umbms.plot.imgplots import plot_fd_img


###############################################################################


def orr_recon(ini_img, freqs, m_size, fd, md, ant_rad, adi_rad, speed,
              n_cores=2,  step_size=0.03, out_dir='', logger=null_logger):
    """Perform optimization-based radar reconstruction, via grad desc

    Parameters
    ----------
    ini_img : array_like
        Initial image estimate
    freqs : array_like
        The frequencies used in the scan
    m_size : int
        The number of pixels along one dimension of the reconstructed image
    fd : array_like
        The measured frequency domain data
    md : dict
        The metadata dictionary for this scan
    ant_rad : float
        The antenna radius, after correcting for the antenna time delay
    adi_rad : float
        The approximate radius of the adipose phantom
    speed : float
        The estimated propagation speed
    n_cores : int
        The number of cores to use for parallel processing
    step_size : float
        The step size to use for gradient descent
    out_dir : str
        The output directory, where the figures and image estimates will
        be saved
    logger :
        Logging object

    Returns
    -------
    img : array_like
        Reconstructed image
    """

    # Get tumour position and size in m
    tum_x = md['tum_x'] / 100
    tum_y = md['tum_y'] / 100
    tum_rad = md['tum_diam'] / 200

    # Get the radius of the region of interest
    roi_rad = adi_rad + 0.01

    # Get the area of each individual pixel
    dv = ((2 * roi_rad)**2) / (m_size**2)

    # Get one-way pixel response times of model
    pix_ts = get_pix_ts(ant_rad=ant_rad, m_size=m_size, roi_rad=roi_rad,
                        speed=speed)

    # Get the phase factor for more efficient computation
    phase_fac = get_fd_phase_factor(pix_ts)

    # Plot the original data in the time and frequency domains
    plt_sino(fd=fd, title="Experimental Data",
             save_str='experimental_data.png',
             close=True, out_dir=out_dir, transparent=False)
    plt_fd_sino(fd=fd, title='Experimental Data',
                save_str='expt_data_fd.png',
                close=True, out_dir=out_dir, transparent=False)

    img = ini_img  # Initialize the image

    # Plot the initial image estimate
    plot_fd_img(np.real(img), tum_x=tum_x, tum_y=tum_y, tum_rad=tum_rad,
                adi_rad=adi_rad, roi_rad=roi_rad,
                img_rad=roi_rad,
                title="Image Estimate Step %d" % 0,
                save_str=os.path.join(out_dir,
                                      "imageEstimate_step_%d.png" % 0),
                save_fig=True,
                save_close=True,
                cbar_fmt='%.2e',
                transparent=False)

    cost_funcs = []  # Init list for storing cost function values

    # Forward project the current image estimate
    fwd = fd_fwd_proj(model=img, phase_fac=phase_fac, dv=dv,
                      n_cores=n_cores,
                      freqs=freqs)

    img_estimates = []  # Init list for storing image estimates

    # Store the initial cost function value
    cost_funcs.append(float(np.sum(np.abs(fwd - fd)**2)))

    # Initialize the number of steps performed in gradient descent
    step = 0

    # Initialize the relative change in the cost function
    cost_rel_change = 1

    logger.info('\tInitial cost value:\t%.4f' % cost_funcs[0])

    # Perform gradient descent until the relative change in the cost
    # function is less than 0.1%
    while cost_rel_change > 0.001:

        logger.info('\tStep %d...' % (step + 1))

        # Plot the forward projection of the image estimate
        plt_sino(fd=fwd, title='Forward Projection Step %d' % (step + 1),
                 save_str='fwdProj_step_%d.png' % (step + 1),
                 close=True, out_dir=out_dir, transparent=False)
        plt_fd_sino(fd=fwd, title='Forward Projection Step %d' % (step + 1),
                    save_str='fwdProj_FD_step_%d.png' % (step + 1),
                    close=True, out_dir=out_dir, transparent=False)

        # Plot the diff between forward and expt data
        plt_sino(fd=(fd - fwd), title='Exp - Fwd Step %d' % (step + 1),
                 save_str='fwdExpDiff_step_%d.png' % (step + 1),
                 close=True, out_dir=out_dir, transparent=False)
        plt_fd_sino(fd=(fd - fwd), title='Exp - Fwd Step %d' % (step + 1),
                    save_str='fwdExpDiff_FD_step_%d.png' % (step + 1),
                    close=True, out_dir=out_dir, transparent=False)

        # Calculate the gradient of the loss function wrt the
        # reflectivities in the object space
        ref_derivs = get_ref_derivs(phase_fac=phase_fac, fd=fd, fwd=fwd,
                                    freqs=freqs, n_cores=n_cores)

        # Update image estimate
        img -= step_size * np.real(ref_derivs)

        # Store the updated image estimate
        img_estimates.append(img * np.ones_like(img))

        # Plot the map of the loss function derivative with respect to
        # each reflectivity point
        plot_fd_img(np.real(ref_derivs), tum_x=tum_x, tum_y=tum_y,
                    tum_rad=tum_rad,
                    adi_rad=adi_rad, roi_rad=roi_rad,
                    img_rad=roi_rad,
                    title="Full Deriv Step %d" % (step + 1),
                    save_str=os.path.join(out_dir,
                                          "fullDeriv_step_%d.png"
                                          % (step + 1)),
                    save_fig=True,
                    save_close=True,
                    cbar_fmt='%.2e',
                    transparent=False)

        # Plot the new image estimate
        plot_fd_img(np.real(img), tum_x=tum_x, tum_y=tum_y,
                    tum_rad=tum_rad,
                    adi_rad=adi_rad, roi_rad=roi_rad,
                    img_rad=roi_rad,
                    title="Image Estimate Step %d" % (step + 1),
                    save_str=os.path.join(out_dir,
                                          "imageEstimate_step_%d.png"
                                          % (step + 1)),
                    save_fig=True,
                    save_close=True,
                    cbar_fmt='%.2e',
                    transparent=False)
        plot_fd_img(np.abs(img), tum_x=tum_x, tum_y=tum_y,
                    tum_rad=tum_rad,
                    adi_rad=adi_rad, roi_rad=roi_rad,
                    img_rad=roi_rad,
                    title="Image Estimate Step %d" % (step + 1),
                    save_str=os.path.join(out_dir,
                                          "imageEstimate_step_%d_abs.png"
                                          % (step + 1)),
                    save_fig=True,
                    save_close=True,
                    cbar_fmt='%.2e',
                    transparent=False)

        # Forward project the current image estimate
        fwd = fd_fwd_proj(model=img, phase_fac=phase_fac, dv=dv,
                          n_cores=n_cores,
                          freqs=freqs)

        # Normalize the forward projection
        cost_funcs.append(np.sum(np.abs(fwd - fd) ** 2))

        logger.info('\t\tCost func:\t%.4f' % (cost_funcs[step + 1]))

        # Calculate the relative change in the cost function
        cost_rel_change = ((cost_funcs[step] - cost_funcs[step + 1])
                           / cost_funcs[step])
        logger.info('\t\t\tCost Func ratio:\t%.4f%%'
                    % (100 * cost_rel_change))

        if step >= 1:  # For each step after the 0th

            # Plot the value of the cost function vs the number of
            # gradient descent steps performed
            plt.figure(figsize=(12, 6))
            plt.rc('font', family='Times New Roman')
            plt.tick_params(labelsize=18)
            plt.plot(np.arange(1, step + 2), cost_funcs[:step + 1],
                     'ko--')
            plt.xlabel('Iteration Number', fontsize=22)
            plt.ylabel('Cost Function Value', fontsize=22)
            plt.title("Optimization Performance with Gradient Descent",
                      fontsize=24)
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(out_dir,
                                     "costFuncs_step_%d.png" % (
                                             step + 1)),
                        transparent=False,
                        dpi=300)
            plt.close()

        step += 1  # Increment the step counter

    # After completing image reconstruction, plot the learning curve
    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=18)
    plt.plot(np.arange(1, len(cost_funcs) + 1), cost_funcs, 'ko--')
    plt.xlabel('Iteration Number', fontsize=22)
    plt.ylabel('Cost Function Value', fontsize=22)
    plt.title("Optimization Performance with Gradient Descent",
              fontsize=24)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, "costFuncs.png"),
                transparent=True,
                dpi=300)
    plt.close()

    # Save the image estimates to a .pickle file
    save_pickle(img_estimates, os.path.join(out_dir, 'img_estimates.pickle'))

    return img


###############################################################################


def fd_das(fd_data, phase_fac, freqs, n_cores=2):
    """Compute frequency-domain DAS reconstruction

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension
    freqs : array_like, Nx1
        The frequencies used in the scan
    n_cores : int
        Number of cores used for parallel processing

    Returns
    -------
    img : array_like, KxK
        Reconstructed image, K pixels by K pixels
    """

    n_fs = np.size(freqs)  # Find number of frequencies used

    # Correct for to/from propagation
    new_phase_fac = phase_fac**(-2)

    # Create func for parallel computation
    parallel_func = partial(_parallel_fd_das_func, fd_data, new_phase_fac,
                            freqs)

    workers = mp.Pool(n_cores)  # Init worker pool

    iterable_idxs = range(n_fs)  # Indices to iterate over

    # Store projections from parallel processing
    back_projections = np.array(workers.map(parallel_func, iterable_idxs))

    # Reshape
    back_projections = np.reshape(back_projections,
                                  [n_fs, np.size(phase_fac, axis=1),
                                   np.size(phase_fac, axis=2)])

    workers.close()  # Close worker pool

    # Sum over all frequencies
    img = np.sum(back_projections, axis=0)

    return img


def _parallel_fd_das_func(fd_data, new_phase_fac, freqs, ff):
    """Compute projection for given frequency ff

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    new_phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension, corrected for DAS
    ff : int
        Frequency index

    Returns
    -------
    this_projection : array_like, KxK
        Back-projection of this particular frequency-point
    """

    # Get phase factor for this frequency
    this_phase_fac = new_phase_fac ** freqs[ff]

    # Sum over antenna positions
    this_projection = np.sum(this_phase_fac * fd_data[ff, :, None, None],
                             axis=0)

    return this_projection


###############################################################################


def fd_dmas(fd_data, pix_ts, freqs):
    """Compute frequency-domain DAS reconstruction

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    pix_ts : array_like
        One-way response times for each pixel in the domain, for each
        antenna position
    freqs : array_like
        The frequencies used in the scan

    Returns
    -------
    img : array_like, KxK
        Reconstructed image, K pixels by K pixels
    """

    # Init array for storing the individual back-projections, from
    # each antenna position
    back_projections = np.zeros([72, np.size(pix_ts, axis=1),
                                 np.size(pix_ts, axis=2)], dtype=complex)

    # For each antenna position
    for aa in range(72):

        # Get the value to back-project
        back_proj_val = (fd_data[:, aa, None, None]
                         * np.exp(-2j * np.pi * freqs[:, None, None]
                                  * (-2 * pix_ts[aa, :, :])))
        # Sum over all frequencies
        back_proj_val = np.sum(back_proj_val, axis=0)

        # Store the back projection
        back_projections[aa, :, :] = back_proj_val

    # Init image to return
    img = np.zeros([np.size(pix_ts, axis=1), np.size(pix_ts, axis=1)],
                   dtype=complex)

    # Loop over each antenna positino
    for aa in range(72):

        # For each other antenna position
        for aa_2 in range(aa + 1, 72):

            img += (back_projections[aa, :, :] * back_projections[aa_2, :, :])

    return img
