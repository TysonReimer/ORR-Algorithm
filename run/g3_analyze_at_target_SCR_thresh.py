"""
Tyson Reimer
University of Manitoba
August 30th, 2021
"""

import os
import numpy as np

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.beamform.iqms import get_scr, get_loc_err, get_scr_healthy
from umbms.beamform.extras import apply_ant_t_delay

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/g3/')

__OUT_DIR = os.path.join(get_proj_path(), 'output/g3/')
verify_path(__OUT_DIR)

# Scan frequency parameters
__INI_F = 1e9
__FIN_F = 9e9
__N_FS = 1001

# Image size
__M_SIZE = 150

# Approximate radius of each adipose shell in our array
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

# The SCR threshold used to determine if a reconstruction is labeled
# as containing a tumour response
__SCR_THRESHOLD = 1.5

# Assumed tumour radius for healthy reconstructions where the SCR
# threshold is exceeded
__HEALTHY_RAD = 0.015

# The str indicating the reference type for the tumour-containing scans
# must be in ['adi', 'fib']
__TUM_REF_STR = 'fib'

###############################################################################

# Define RGB colours for plotting DAS/DMAS/ORR
das_col = [0, 0, 0]
dmas_col = [80, 80, 80]
gd_col = [160, 160, 160]
das_col = [ii / 255 for ii in das_col]
dmas_col = [ii / 255 for ii in dmas_col]
gd_col = [ii / 255 for ii in gd_col]

###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load the metadata for all scans
    metadata = load_pickle(os.path.join(__DATA_DIR,
                                        'metadata_gen_three.pickle'))

    n_expts = len(metadata)  # Find number of experiments / scans

    # Scan freqs and target freqs
    scan_fs = np.linspace(__INI_F, __FIN_F, __N_FS)

    # Retain only frequencies above 2 GHz, due to antenna VSWR
    tar_fs = scan_fs >= 2e9

    # The directory where reconstructed images are stored
    img_dir = os.path.join(__OUT_DIR, 'recons/')

    # Init list for storing all metadata
    all_md = []

    # Init lists for storing the SCR and localization errors, for
    # each beamforming method
    das_scrs = []
    das_les = []
    dmas_scrs = []
    dmas_les = []
    orr_scrs = []
    orr_les = []

    # Init lists for storing the detects and healthy detects for
    # each beamformer
    das_detects = []
    dmas_detects = []
    orr_detects = []
    das_healthy_detects = []
    dmas_healthy_detects = []
    orr_healthy_detects = []

    # Make output dir for figures
    fig_out_dir = os.path.join(__OUT_DIR, 'iqms/')
    verify_path(fig_out_dir)

    # For each experiment / scan
    for ii in range(n_expts):

        # Get the metadata for this scan
        tar_md = metadata[ii]

        # If the scan had a fibroglandular shell (indicating it was of
        # a complete tumour-containing or healthy phantom)
        if 'F' in tar_md['phant_id'] and ~np.isnan(tar_md['tum_diam']):

            # Use the fibroglandular reference scan
            tar_img_dir = os.path.join(img_dir, 'id-%d-%s/'
                                       % (tar_md['id'], __TUM_REF_STR))

            # Load the ORR reconstructions (at each step)
            orr_imgs = load_pickle(os.path.join(tar_img_dir,
                                                'img_estimates.pickle'))
            orr_img = orr_imgs[-1]  # Examine the final image

            # Load the DAS and DMAS reconstructions
            das_img = load_pickle(os.path.join(tar_img_dir,
                                               'das_%s.pickle'
                                               % __TUM_REF_STR))
            dmas_img = load_pickle(os.path.join(tar_img_dir,
                                                'dmas_%s.pickle'
                                                % __TUM_REF_STR))

            # Get metadata for plotting
            scan_rad = tar_md['ant_rad'] / 100
            tum_x = tar_md['tum_x'] / 100
            tum_y = tar_md['tum_y'] / 100
            tum_rad = 0.5 * (tar_md['tum_diam'] / 100)
            adi_rad = __ADI_RADS[tar_md['phant_id'].split('F')[0]]

            # Correct for the antenna radius measurement position
            # (measured from point on antenna stand, not from SMA
            # connection location)
            scan_rad += 0.03618

            # Define the radius of the region of interest
            roi_rad = adi_rad + 0.01

            # Correct for the antenna time delay
            ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

            # Get the SCR and localization error for the DAS image
            das_scr, das_d_scr = get_scr(img=das_img, roi_rad=roi_rad,
                                         adi_rad=adi_rad,
                                         tum_rad=tum_rad,
                                         tum_x=tum_x, tum_y=tum_y)
            das_le = get_loc_err(img=das_img, ant_rad=roi_rad,
                                 tum_x=tum_x, tum_y=tum_y)

            # Get the SCR and localization error for the DMAS image
            dmas_scr, dmas_d_scr = get_scr(img=dmas_img, roi_rad=roi_rad,
                                           adi_rad=adi_rad,
                                           tum_rad=tum_rad,
                                           tum_x=tum_x, tum_y=tum_y)
            dmas_le = get_loc_err(img=dmas_img, ant_rad=roi_rad,
                                  tum_x=tum_x, tum_y=tum_y)

            # Get the SCR and localization error for the ORR image
            orr_scr, orr_d_scr = get_scr(img=orr_img, roi_rad=roi_rad,
                                         adi_rad=adi_rad,
                                         tum_rad=tum_rad,
                                         tum_x=tum_x, tum_y=tum_y)
            orr_le = get_loc_err(img=orr_img, ant_rad=roi_rad,
                                 tum_x=tum_x, tum_y=tum_y)

            # Store the results
            das_scrs.append((das_scr, das_d_scr))
            das_les.append(das_le)
            dmas_scrs.append((dmas_scr, dmas_d_scr))
            dmas_les.append(dmas_le)
            orr_scrs.append((orr_scr, orr_d_scr))
            orr_les.append(orr_le)

            # Use the tumour detection criteria to determine if
            # a tumour was *accurately* (i.e., the 'detected tumor'
            # corresponds to the true tumor) detected in the
            # reconstructions
            das_detect = (das_scr >= __SCR_THRESHOLD
                          and das_le <= (tum_rad + 0.005))
            dmas_detect = (dmas_scr >= __SCR_THRESHOLD
                           and dmas_le <= (tum_rad + 0.005))
            orr_detect = (orr_scr >= __SCR_THRESHOLD
                          and orr_le <= (tum_rad + 0.005))

            # Store the true detection results
            orr_detects.append(orr_detect)
            das_detects.append(das_detect)
            dmas_detects.append(dmas_detect)

            all_md.append(tar_md)

        # If the experiment was of a healthy phantom
        elif 'F' in tar_md['phant_id'] and np.isnan(tar_md['tum_diam']):

            # Get the directory for this image
            tar_img_dir = os.path.join(img_dir, 'id-%d-adi/'
                                       % tar_md['id'])

            # Load the ORR reconstructions (at each step)
            orr_imgs = load_pickle(os.path.join(tar_img_dir,
                                                'img_estimates.pickle'))
            orr_img = orr_imgs[-1]  # Examine final reconstruction

            # Load the DAS and DMAS reconstructions
            das_img = load_pickle(os.path.join(tar_img_dir,
                                               'das_adi.pickle'))
            dmas_img = load_pickle(os.path.join(tar_img_dir,
                                                'dmas_adi.pickle'))

            # Get metadata for plotting
            scan_rad = tar_md['ant_rad'] / 100
            tum_x = tar_md['tum_x'] / 100
            tum_y = tar_md['tum_y'] / 100
            tum_rad = 0.5 * (tar_md['tum_diam'] / 100)
            adi_rad = __ADI_RADS[tar_md['phant_id'].split('F')[0]]

            # Correct for the antenna radius measurement position
            # (measured from point on antenna stand, not from SMA
            # connection location)
            scan_rad += 0.03618

            # Define the region of interest
            roi_rad = adi_rad + 0.01

            # Correct for the antenna time delay
            ant_rad = apply_ant_t_delay(scan_rad=scan_rad, new_ant=True)

            # Get the SCR for DAS
            das_scr, das_d_scr = get_scr_healthy(img=np.abs(das_img),
                                                 roi_rad=roi_rad,
                                                 adi_rad=adi_rad,
                                                 ant_rad=roi_rad,
                                                 healthy_rad=__HEALTHY_RAD)

            # Get the SCR for DMAS
            dmas_scr, dmas_d_scr = get_scr_healthy(img=np.abs(dmas_img),
                                                   roi_rad=roi_rad,
                                                   adi_rad=adi_rad,
                                                   ant_rad=roi_rad,
                                                   healthy_rad=__HEALTHY_RAD)

            # Get the SCR for ORR
            orr_scr, orr_d_scr = get_scr_healthy(img=np.abs(orr_img),
                                                 roi_rad=roi_rad,
                                                 adi_rad=adi_rad,
                                                 ant_rad=roi_rad,
                                                 healthy_rad=__HEALTHY_RAD)

            # Determine if a tumour was detected in each image
            das_detect = das_scr >= __SCR_THRESHOLD
            dmas_detect = dmas_scr >= __SCR_THRESHOLD
            orr_detect = orr_scr >= __SCR_THRESHOLD

            # Store the detection results
            das_healthy_detects.append(das_detect)
            dmas_healthy_detects.append(dmas_detect)
            orr_healthy_detects.append(orr_detect)

    # Calculate and store the sensitivities
    das_sensitivity = (100 * np.sum(das_detects) / len(das_detects))
    dmas_sensitivity = (100 * np.sum(dmas_detects) / len(dmas_detects))
    orr_sensitivity = (100 * np.sum(orr_detects) / len(orr_detects))
    # Calculate and store the specificities
    das_specificity = (100 * np.sum(1 - np.array(das_healthy_detects))
                       / len(das_healthy_detects))
    dmas_specificity = (100 * np.sum(1 - np.array(dmas_healthy_detects))
                        / len(dmas_healthy_detects))
    orr_specificity = (100 * np.sum(1 - np.array(orr_healthy_detects))
                       / len(orr_healthy_detects))

    # Report the sensitivities and specificities at this SCR
    # threshold to the logger
    logger.info('--------------------------------------------------------')
    # Report DAS
    logger.info('\tDAS Sensitivity:\t%.2f%%' % das_sensitivity)
    logger.info('\tDAS Specificity:\t%.2f%%' % das_specificity)
    # Report DMAS
    logger.info('\tDMAS Sensitivity:\t%.2f%%' % dmas_sensitivity)
    logger.info('\tDMAS Specificity:\t%.2f%%' % dmas_specificity)
    # Report ORR
    logger.info('\tORR Sensitivity:\t\t%.2f%%' % orr_sensitivity)
    logger.info('\tORR Specificity:\t\t%.2f%%' % orr_specificity)

    save_pickle((das_sensitivity, dmas_sensitivity, orr_sensitivity),
                os.path.join(__OUT_DIR,
                             '%s_sensitivities_at_target_threshold.pickle'
                             % __TUM_REF_STR))
    save_pickle((das_specificity, dmas_specificity, orr_specificity),
                os.path.join(__OUT_DIR,
                             '%s_specificities_at_target_threshold.pickle'
                             % __TUM_REF_STR))

    # Do analysis based on tumour size...
    # Define the tumour sizes (in cm)
    tum_sizes = [3, 2.5, 2, 1.5, 1]

    das_tum_sens = np.zeros_like(tum_sizes)
    dmas_tum_sens = np.zeros_like(tum_sizes)
    orr_tum_sens = np.zeros_like(tum_sizes)

    for ii in range(len(tum_sizes)):  # For each tumour size

        # Find the expts that had this tumour size
        tar_idxs = [md['tum_diam'] == tum_sizes[ii] for md in all_md]

        # Find the detection results here
        das_detects_here = np.array(das_detects)[tar_idxs]
        dmas_detects_here = np.array(dmas_detects)[tar_idxs]
        orr_detects_here = np.array(orr_detects)[tar_idxs]

        # Calculate the sensitivities
        das_sens = np.sum(das_detects_here) * 100 / len(das_detects_here)
        dmas_sens = np.sum(dmas_detects_here) * 100 / len(dmas_detects_here)
        orr_sens = np.sum(orr_detects_here) * 100 / len(orr_detects_here)

        das_tum_sens[ii] = das_sens
        dmas_tum_sens[ii] = dmas_sens
        orr_tum_sens[ii] = orr_sens

        # Report results
        logger.info('\t%.1f cm Tumours:' % ii)
        logger.info('\t\tGD Sensitivity:\t\t%.2f%%' % orr_sens)
        logger.info('\t\tDAS Sensitivity:\t%.2f%%' % das_sens)
        logger.info('\t\tDMAS Sensitivity:\t%.2f%%' % dmas_sens)

    save_pickle((das_tum_sens, dmas_tum_sens, orr_tum_sens),
                os.path.join(__OUT_DIR, '%s_sens_by_tums.pickle'
                             % __TUM_REF_STR))
