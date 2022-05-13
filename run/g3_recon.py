"""
Tyson Reimer
University of Manitoba
August 30th, 2021
"""

import os
import numpy as np

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.plot.imgplots import plot_fd_img

from umbms.beamform.recon import fd_das, fd_dmas, orr_recon
from umbms.beamform.extras import (apply_ant_t_delay, get_pix_ts,
                                   get_fd_phase_factor)
from umbms.beamform.propspeed import estimate_speed

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/g3/')

__OUT_DIR = os.path.join(get_proj_path(), 'output/g3/')
verify_path(__OUT_DIR)

# The frequency parameters from the scan
__INI_F = 1e9
__FIN_F = 9e9
__N_FS = 1001

# The time-domain conversion parameters
__INI_T = 0.5e-9
__FIN_T = 5.5e-9
__N_TS = 700

# The size of the reconstructed image along one dimension
__M_SIZE = 150

# The approximate radius of each adipose phantom in our array
__ADI_RADS = {
    'A1': 0.05,
    'A2': 0.06,
    'A3': 0.07,
    'A11': 0.06,
    'A12': 0.05,
    'A13': 0.065,
    'A14': 0.06,
    'A15': 0.055,
    'A16': 0.07
}

###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load the frequency domain data and metadata
    fd_data = load_pickle(os.path.join(__DATA_DIR,
                                       'fd_data_gen_three_s11.pickle'))
    metadata = load_pickle(os.path.join(__DATA_DIR,
                                        'metadata_gen_three.pickle'))

    n_expts = len(metadata)  # The number of individual scans

    # Get the unique ID of each experiment / scan
    expt_ids = [md['id'] for md in metadata]

    # Get the unique IDs of the adipose-only and adipose-fibroglandular
    # (healthy) reference scans for each experiment/scan
    adi_ref_ids = [md['adi_ref_id'] for md in metadata]
    fib_ref_ids = [md['fib_ref_id'] for md in metadata]

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    # Only retain frequencies above 2 GHz due to antenna VSWR
    tar_fs = scan_fs >= 2e9

    # The output dir, where the reconstructions will be stored
    out_dir = os.path.join(__OUT_DIR, 'recons/')
    verify_path(out_dir)

    # for ii in range(n_expts):  # For each scan / experiment
    for ii in [1]:

        logger.info('Scan [%3d / %3d]...' % (ii + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = fd_data[ii, :, :]
        tar_md = metadata[ii]
        print(tar_md['id'])

        # If the scan had a fibroglandular shell (indicating it was of
        # a complete tumour-containing or healthy phantom)
        if 'F' in tar_md['phant_id']:

            # Create the output directory for the adipose-only
            # reference reconstructions
            expt_adi_out_dir = os.path.join(out_dir, 'id-%d-adi/'
                                            % tar_md['id'])
            verify_path(expt_adi_out_dir)

            # Create the output directory for the adipose-fibroglandular
            # reference reconstructions
            expt_fib_out_dir = os.path.join(out_dir, 'id-%d-fib/'
                                            % tar_md['id'])
            verify_path(expt_fib_out_dir)

            # Get metadata for plotting
            scan_rad = tar_md['ant_rad'] / 100
            tum_x = tar_md['tum_x'] / 100
            tum_y = tar_md['tum_y'] / 100
            tum_rad = 0.5 * (tar_md['tum_diam'] / 100)
            adi_rad = __ADI_RADS[tar_md['phant_id'].split('F')[0]]

            # Correct for how the scan radius is measured (from a
            # point on the antenna stand, not at the SMA connection
            # point)
            scan_rad += 0.03618

            # Define the radius of the region of interest
            roi_rad = adi_rad + 0.01

            # Get the area of each pixel in the image domain
            dv = ((2 * roi_rad) ** 2) / (__M_SIZE ** 2)

            # Correct for the antenna time delay
            # NOTE: Only the new antenna was used in UM-BMID Gen-3
            ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

            # Estimate the propagation speed in the imaging domain
            speed = estimate_speed(adi_rad=adi_rad, ant_rad=scan_rad,
                                   new_ant=True)

            # Get the one-way propagation times for each pixel,
            # for each antenna position
            pix_ts = get_pix_ts(ant_rad=ant_rad, m_size=__M_SIZE,
                                roi_rad=roi_rad, speed=speed)

            # Get the phase factor for efficient computation
            phase_fac = get_fd_phase_factor(pix_ts=pix_ts)

            # Get the adipose-only reference data for this scan
            adi_fd = fd_data[expt_ids.index(tar_md['adi_ref_id']), :, :]

            # Subtract reference and retain the frequencies above 2 GHz
            adi_cal_cropped = (tar_fd - adi_fd)[tar_fs, :]

            # If the scan does include a tumour
            if ~np.isnan(tar_md['tum_diam']):

                # Set a str for plotting
                plt_str = "%.1f cm tum in " \
                          "Class %d %s, ID: %d" % (tar_md['tum_diam'],
                                                   tar_md['birads'],
                                                   tar_md['phant_id'],
                                                   tar_md['id'])

            else:  # If the scan does NOT include a tumour
                plt_str = "Class %d %s, ID: %d" % (tar_md['birads'],
                                                   tar_md['phant_id'],
                                                   tar_md['id'])

            # Reconstruct a DAS image
            das_adi_recon = fd_das(fd_data=adi_cal_cropped,
                                   phase_fac=phase_fac,
                                   freqs=scan_fs[tar_fs], n_cores=4)

            # Save that DAS reconstruction to a .pickle file
            save_pickle(das_adi_recon,
                        os.path.join(expt_adi_out_dir, 'das_adi.pickle'))

            # Plot the DAS reconstruction
            plot_fd_img(img=np.abs(das_adi_recon), tum_x=tum_x, tum_y=tum_y,
                        tum_rad=tum_rad, adi_rad=adi_rad,
                        ant_rad=ant_rad, roi_rad=roi_rad,
                        img_rad=roi_rad,
                        title='%s\nAdi Cal' % plt_str,
                        save_fig=True,
                        save_str=os.path.join(out_dir,
                                              'id_%d_adi_cal_das.png'
                                              % tar_md['id']),
                        save_close=True)

            # Reconstruct a DMAS image
            dmas_adi_recon = fd_dmas(fd_data=adi_cal_cropped, pix_ts=pix_ts,
                                     freqs=scan_fs[tar_fs])

            # Save the DMAS reconstruction to a .pickle file
            save_pickle(dmas_adi_recon,
                        os.path.join(expt_adi_out_dir, 'dmas_adi.pickle'))

            # Plot the DMAS reconstruction
            plot_fd_img(img=np.abs(dmas_adi_recon), tum_x=tum_x, tum_y=tum_y,
                        tum_rad=tum_rad, adi_rad=adi_rad,
                        ant_rad=ant_rad, roi_rad=roi_rad,
                        img_rad=roi_rad,
                        title='%s\nAdi Cal' % plt_str,
                        save_fig=True,
                        save_str=os.path.join(out_dir,
                                              'id_%d_adi_cal_dmas.png'
                                              % tar_md['id']),
                        save_close=True)

            # Reconstruct an ORR image
            adi_orr = orr_recon(ini_img=np.real(das_adi_recon),
                                freqs=scan_fs[tar_fs],
                                m_size=__M_SIZE,
                                fd=adi_cal_cropped,
                                md=tar_md,
                                adi_rad=adi_rad,
                                ant_rad=ant_rad,
                                speed=speed,
                                n_cores=4,
                                out_dir=expt_adi_out_dir,
                                logger=logger)

            # Plot the ORR image
            plot_fd_img(img=np.abs(adi_orr), tum_x=tum_x, tum_y=tum_y,
                        tum_rad=tum_rad, adi_rad=adi_rad,
                        ant_rad=ant_rad, roi_rad=roi_rad,
                        img_rad=roi_rad,
                        title='%s\nAdi Cal' % plt_str,
                        save_fig=True,
                        save_str=os.path.join(out_dir,
                                              'id_%d_adi_cal_orr.png'
                                              % tar_md['id']),
                        save_close=True)

            # If the scan contained a tumour
            if ~np.isnan(tar_md['tum_diam']):

                # Get the data for the adipose-fibroglandular reference
                # scan
                fib_fd = fd_data[expt_ids.index(tar_md['fib_ref_id']), :, :]

                # Subtract reference and crop frequencies
                fib_cal_cropped = (tar_fd - fib_fd)[tar_fs, :]

                # Produce DAS reconstruction, save, plot
                das_fib_recon = fd_das(fd_data=fib_cal_cropped,
                                       phase_fac=phase_fac,
                                       freqs=scan_fs[tar_fs],
                                       n_cores=4)
                save_pickle(das_fib_recon,
                            os.path.join(expt_fib_out_dir, 'das_fib.pickle'))
                plot_fd_img(img=np.abs(das_fib_recon), tum_x=tum_x,
                            tum_y=tum_y,
                            tum_rad=tum_rad, adi_rad=adi_rad,
                            ant_rad=ant_rad, roi_rad=roi_rad,
                            img_rad=roi_rad,
                            title='%s\nFib Cal' % plt_str,
                            save_fig=True,
                            save_str=os.path.join(out_dir,
                                                  'id_%d_fib_cal_das.png'
                                                  % tar_md['id']),
                            save_close=True)

                # Produce DMAS reconstruction, save, plot
                dmas_fib_recon = fd_dmas(fd_data=fib_cal_cropped,
                                         pix_ts=pix_ts, freqs=scan_fs[tar_fs])
                save_pickle(dmas_fib_recon,
                            os.path.join(expt_fib_out_dir, 'dmas_fib.pickle'))
                plot_fd_img(img=np.abs(dmas_fib_recon), tum_x=tum_x,
                            tum_y=tum_y,
                            tum_rad=tum_rad, adi_rad=adi_rad,
                            ant_rad=ant_rad, roi_rad=roi_rad,
                            img_rad=roi_rad,
                            title='%s\nAdi Cal' % plt_str,
                            save_fig=True,
                            save_str=os.path.join(out_dir,
                                                  'id_%d_fib_cal_dmas.png'
                                                  % tar_md['id']),
                            save_close=True)

                # Produce ORR reconstruction, save, plot
                fib_gd = orr_recon(ini_img=np.real(das_fib_recon),
                                   freqs=scan_fs[tar_fs],
                                   m_size=__M_SIZE,
                                   fd=fib_cal_cropped,
                                   md=tar_md,
                                   adi_rad=adi_rad,
                                   ant_rad=ant_rad,
                                   speed=speed,
                                   n_cores=4,
                                   out_dir=expt_fib_out_dir,
                                   logger=logger)
                plot_fd_img(img=np.abs(fib_gd), tum_x=tum_x, tum_y=tum_y,
                            tum_rad=tum_rad, adi_rad=adi_rad,
                            ant_rad=ant_rad, roi_rad=roi_rad,
                            img_rad=roi_rad,
                            title='%s\nFib Cal' % plt_str,
                            save_fig=True,
                            save_str=os.path.join(out_dir,
                                                  'id_%d_fib_cal_gd.png'
                                                  % tar_md['id']),
                            save_close=True)
