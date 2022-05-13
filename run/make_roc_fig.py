"""
Tyson Reimer
University of Manitoba
November 15th, 2021
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from umbms import get_proj_path
from umbms.loadsave import load_pickle

###############################################################################

__DIR = os.path.join(get_proj_path(), 'output/g3/')

# Define colors for plotting
das_col = [0, 0, 0]
dmas_col = [80, 80, 80]
gd_col = [160, 160, 160]
das_col = [ii / 255 for ii in das_col]
dmas_col = [ii / 255 for ii in dmas_col]
gd_col = [ii / 255 for ii in gd_col]

# Define additional colors for plotting
das_col2 = [124, 0, 0]
dmas_col2 = [255, 0, 0]
gd_col2 = [231, 167, 167]
das_col2 = [ii / 255 for ii in das_col2]
dmas_col2 = [ii / 255 for ii in dmas_col2]
gd_col2 = [ii / 255 for ii in gd_col2]

###############################################################################


def get_aucs():
    ####### 2022-03-01 stuff....

    das_tps_adi = (das_adi_sens) / 100
    das_fps_adi = (100 - das_adi_spec) / 100
    dmas_tps_adi = (dmas_adi_sens) / 100
    dmas_fps_adi = (100 - dmas_adi_spec) / 100
    gd_tps_adi = (gd_adi_sens) / 100
    gd_fps_adi = (100 - gd_adi_spec) / 100

    das_tps_fib = (das_fib_sens) / 100
    das_fps_fib = (100 - das_fib_spec) / 100
    dmas_tps_fib = (dmas_fib_sens) / 100
    dmas_fps_fib = (100 - dmas_fib_spec) / 100
    gd_tps_fib = (gd_fib_sens) / 100
    gd_fps_fib = (100 - gd_fib_spec) / 100

    ##########

    das_adi_sort = np.argsort(das_fps_adi)
    das_tps_adi = das_tps_adi[das_adi_sort]
    das_fps_adi = das_fps_adi[das_adi_sort]
    dmas_adi_sort = np.argsort(dmas_fps_adi)
    dmas_tps_adi = dmas_tps_adi[dmas_adi_sort]
    dmas_fps_adi = dmas_fps_adi[dmas_adi_sort]
    gd_adi_sort = np.argsort(gd_fps_adi)
    gd_tps_adi = gd_tps_adi[gd_adi_sort]
    gd_fps_adi = gd_fps_adi[gd_adi_sort]

    das_fib_sort = np.argsort(das_fps_fib)
    das_tps_fib = das_tps_fib[das_fib_sort]
    das_fps_fib = das_fps_fib[das_fib_sort]
    dmas_fib_sort = np.argsort(dmas_fps_fib)
    dmas_tps_fib = dmas_tps_fib[dmas_fib_sort]
    dmas_fps_fib = dmas_fps_fib[dmas_fib_sort]
    gd_fib_sort = np.argsort(gd_fps_fib)
    gd_tps_fib = gd_tps_fib[gd_fib_sort]
    gd_fps_fib = gd_fps_fib[gd_fib_sort]

    das_adi_auc = 0
    dmas_adi_auc = 0
    gd_adi_auc = 0

    das_fib_auc = 0
    dmas_fib_auc = 0
    gd_fib_auc = 0

    for ii in range(len(das_tps_adi) - 1):

        das_d_fps = das_fps_adi[ii + 1] - das_fps_adi[ii]
        dmas_d_fps = dmas_fps_adi[ii + 1] - dmas_fps_adi[ii]
        gd_d_fps = gd_fps_adi[ii + 1] - gd_fps_adi[ii]

        if das_d_fps != 0:
            das_adi_auc += das_d_fps * das_tps_adi[ii + 1]

        if dmas_d_fps != 0:
            dmas_adi_auc += dmas_d_fps * dmas_tps_adi[ii + 1]

        if gd_d_fps != 0:
            gd_adi_auc += gd_d_fps * gd_tps_adi[ii + 1]

        ###

        das_fib_d_fps = das_fps_fib[ii + 1] - das_fps_fib[ii]
        dmas_fib_d_fps = dmas_fps_fib[ii + 1] - dmas_fps_fib[ii]
        gd_fib_d_fps = gd_fps_fib[ii + 1] - gd_fps_fib[ii]

        if das_fib_d_fps != 0:
            das_fib_auc += das_fib_d_fps * das_tps_fib[ii + 1]

        if dmas_fib_d_fps != 0:
            dmas_fib_auc += dmas_fib_d_fps * dmas_tps_fib[ii + 1]

        if gd_fib_d_fps != 0:
            gd_fib_auc += gd_fib_d_fps * gd_tps_fib[ii + 1]

    return (das_adi_auc, dmas_adi_auc, gd_adi_auc,
            das_fib_auc, dmas_fib_auc, gd_fib_auc)



###############################################################################


if __name__ == "__main__":

    # Load the sensitivities and specificities when adipose-only scans
    # are used for reference subtraction
    (_, das_adi_sens, dmas_adi_sens, gd_adi_sens) = \
        load_pickle(os.path.join(__DIR, 'adi_ref_sensitvities.pickle'))
    (_, das_adi_spec, dmas_adi_spec, gd_adi_spec) = \
        load_pickle(os.path.join(__DIR, 'adi_ref_specificities.pickle'))

    # Load the sensitivities and specificities when
    # adipose-fibroglandular (healthy) scans are used for
    # reference subtraction
    (_, das_fib_sens, dmas_fib_sens, gd_fib_sens) = \
        load_pickle(os.path.join(__DIR, 'fib_ref_sensitvities.pickle'))
    (_, das_fib_spec, dmas_fib_spec, gd_fib_spec) = \
        load_pickle(os.path.join(__DIR, 'fib_ref_specificities.pickle'))

    (das_adi_auc, dmas_adi_auc, gd_adi_auc,
     das_fib_auc, dmas_fib_auc, gd_fib_auc) = get_aucs()

    # Make the figure
    plt.figure(figsize=(10, 8))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=(20))
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot the ROC curves when adipose-only references are used
    plt.plot(100 - das_adi_spec, das_adi_sens, c=das_col, linestyle='-',
             label="DAS, Adipose Reference (AUC: %.1f%%)"
                   % (100 * das_adi_auc))
    plt.plot(100 - dmas_adi_spec, dmas_adi_sens, c=dmas_col, linestyle='--',
             label="DMAS, Adipose Reference (AUC: %.1f%%)"
                   % (100 * dmas_adi_auc))
    plt.plot(100 - gd_adi_spec, gd_adi_sens, c=gd_col, linestyle='--',
             label="ORR, Adipose Reference (AUC: %.1f%%)"
                   % (100 * gd_adi_auc))

    # Plot the ROC curves when healthy references are used
    plt.plot(100 - das_fib_spec, das_fib_sens, c=das_col2, linestyle='-',
             label="DAS, Healthy Reference (AUC: %.1f%%)"
                   % (100 * das_fib_auc))
    plt.plot(100 - dmas_fib_spec, dmas_fib_sens, c=dmas_col2, linestyle='--',
             label="DMAS, Healthy Reference (AUC: %.1f%%)"
                   % (100 * dmas_fib_auc))
    plt.plot(100 - gd_fib_spec, gd_fib_sens, c=gd_col2, linestyle='--',
             label="ORR, Healthy Reference (AUC: %.1f%%)"
                   % (100 * gd_fib_auc))

    # Plot the ROC curve of a random classifier
    plt.plot(np.linspace(0, 100, 100), np.linspace(0, 100, 100),
             c=[0, 108 / 255, 255 / 255], linestyle='--',
             label='Random Classifier')

    # Make the legend, axes labels, etc
    plt.legend(fontsize=16, loc='center right')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel("False Positive Rate (%)", fontsize=22)
    plt.ylabel("True Positive Rate (%)", fontsize=22)
    plt.tight_layout()
    plt.show()
    # Save the fig
    # plt.savefig(os.path.join(__DIR, 'rocs.png'), dpi=300,
    #             transparent=True)



