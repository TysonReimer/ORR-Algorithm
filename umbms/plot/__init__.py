"""
Tyson Reimer
University of Manitoba
July 09th, 2020
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from umbms.beamform.sigproc import iczt

###############################################################################


def init_plt(figsize=(12, 6), labelsize=18):
    """

    Parameters
    ----------
    figsize : tuple
        The figure size
    labelsize : int
        The labelsize for the axis-ticks
    """

    plt.figure(figsize=figsize)
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=labelsize)


###############################################################################


def plt_sino(fd, title, save_str, out_dir, cbar_fmt='%.2e',
             transparent=True, close=True):

    # Find the minimum retained frequency
    scan_fs = np.linspace(1e9, 8e9, 1001)  # Frequencies used in scan
    min_f = 2e9  # Min frequency to retain
    tar_fs = scan_fs >= min_f  # Target frequencies to retain
    min_retain_f = np.min(scan_fs[tar_fs])  # Min freq actually retained

    # Create variables for plotting
    ts = np.linspace(0.5, 5.5, 700)
    plt_extent = [0, 355, ts[-1], ts[0]]
    plt_aspect_ratio = 355 / ts[-1]

    # Conert to the time-domain
    td = iczt(fd, ini_t=0.5e-9, fin_t=5.5e-9, n_time_pts=700,
              ini_f=min_retain_f, fin_f=8e9)

    # Plot primary scatter forward projection only
    plt.figure()
    plt.rc('font', family='Times New Roman')
    plt.imshow(np.abs(td), aspect=plt_aspect_ratio, cmap='inferno',
               extent=plt_extent)
    plt.colorbar(format=cbar_fmt).ax.tick_params(labelsize=16)
    plt.gca().set_yticks([round(ii, 2)
                          for ii in ts[::700 // 8]])
    plt.gca().set_xticks([round(ii)
                          for ii in np.linspace(0, 355, 355)[::75]])
    plt.title('%s' % title, fontsize=20)
    plt.xlabel('Polar Angle of Antenna Position ('
               + r'$^\circ$' + ')',
               fontsize=16)
    plt.ylabel('Time of Response (ns)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '%s' % save_str),
                dpi=300, transparent=transparent)
    if close:
        plt.close()


def plt_fd_sino(fd, title, save_str, out_dir, cbar_fmt='%.2e',
                transparent=True, close=True):

    # Find the minimum retained frequency
    scan_fs = np.linspace(1e9, 8e9, 1001)  # Frequencies used in scan
    min_f = 2e9  # Min frequency to retain
    tar_fs = scan_fs >= min_f  # Target frequencies to retain
    min_retain_f = np.min(scan_fs[tar_fs])  # Min freq actually retained

    # Create variables for plotting
    fs = scan_fs[tar_fs]
    plt_extent = [0, 355, fs[-1] / 1e9, fs[0] / 1e9]
    plt_aspect_ratio = 355 / (fs[-1] / 1e9)

    # Plot primary scatter forward projection only
    plt.figure()
    plt.rc('font', family='Times New Roman')
    plt.imshow(np.abs(fd), aspect=plt_aspect_ratio, cmap='inferno',
               extent=plt_extent)
    plt.colorbar(format=cbar_fmt).ax.tick_params(labelsize=16)
    plt.tick_params(labelsize=14)
    plt.gca().set_yticks([2, 3, 4, 5, 6, 7, 8])
    plt.gca().set_xticks([round(ii)
                          for ii in np.linspace(0, 355, 355)[::75]])
    plt.title('%s' % title, fontsize=20)
    plt.xlabel('Polar Angle of Antenna Position ('
               + r'$^\circ$' + ')',
               fontsize=16)
    plt.ylabel('Frequency (GHz)', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '%s' % save_str),
                dpi=300, transparent=transparent)
    if close:
        plt.close()
