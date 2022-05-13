"""
Tyson Reimer
University of Manitoba
September 10th, 2021
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path

from umbms.loadsave import load_pickle

###############################################################################

__OUT_DIR = os.path.join(get_proj_path(), 'output/g3/diagnostic-figs/')
verify_path(__OUT_DIR)

# Define RGB colors for plotting
das_col = [0, 0, 0]
dmas_col = [80, 80, 80]
orr_col = [160, 160, 160]
das_col = [ii / 255 for ii in das_col]
dmas_col = [ii / 255 for ii in dmas_col]
orr_col = [ii / 255 for ii in orr_col]

###############################################################################


def plt_adi_ref_performance():
    """Plot the diagnostic performance when adipose-only used as ref
    """

    # Define x-coords for bars
    das_xs = [0, 3.5]
    dmas_xs = [1, 4.5]
    orr_xs = [2, 5.5]

    # Init plot
    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=20)

    # Plot DAS sensitivity and specificity
    plt.bar(das_xs,
            height=[das_sens[0], das_spec],
            width=0.75,
            linewidth=1,
            color=das_col,
            edgecolor='k',
            label='DAS')

    # Plot DMAS sensitivity and specificity
    plt.bar(dmas_xs,
            height=[dmas_sens[0], dmas_spec],
            width=0.75,
            capsize=10,
            linewidth=1,
            color=dmas_col,
            edgecolor='k',
            label='DMAS')

    # Plot
    plt.bar(orr_xs,
            height=[orr_sens[0], orr_spec],
            width=0.75,
            capsize=10,
            linewidth=1,
            color=orr_col,
            edgecolor='k',
            label='ORR')

    plt.legend(fontsize=18,
               loc='upper left',
               framealpha=0.95)

    plt.xticks([1, 4.5],
               ["Sensitivity", "Specificity"],
               size=20)

    plt.ylabel('Metric Value (%)', fontsize=22)

    # Put text on the bars
    das_text_ys = [15, 40]
    das_for_text = [das_sens[0], das_spec]
    dmas_text_ys = [16, 36]
    dmas_for_text = [dmas_sens[0], dmas_spec]
    gd_text_ys = [23, 52]
    gd_for_text = [orr_sens[0], orr_spec]

    for ii in range(len(das_text_ys)):
        plt.text(das_xs[ii], das_text_ys[ii],
                 "%d" % das_for_text[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    for ii in range(len(dmas_text_ys)):
        plt.text(dmas_xs[ii], dmas_text_ys[ii],
                 "%d" % dmas_for_text[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    for ii in range(len(gd_text_ys)):
        plt.text(orr_xs[ii], gd_text_ys[ii],
                 "%d" % gd_for_text[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    # Set appropriate y-limit
    plt.ylim([0, 100])
    plt.tight_layout()  # Make everything fit nicely
    plt.show()  # Display the plot

    # Save the figure
    plt.savefig(os.path.join(__OUT_DIR, 'sens_spec_adi_refs.png'),
                dpi=300, transparent=False)


def plt_fib_ref_performance():
    """Plot the diagnostic performance when healthy scan as reference
    """

    # Define x-coords for bars
    das_xs = [0, 3.5]
    dmas_xs = [1, 4.5]
    orr_xs = [2, 5.5]

    # Init fig
    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=20)

    plt.bar(das_xs,
            height=[das_sens[1], das_spec],
            width=0.75,
            linewidth=1,
            color=das_col,
            edgecolor='k',
            label='DAS')

    plt.bar(dmas_xs,
            height=[dmas_sens[1], dmas_spec],
            width=0.75,
            capsize=10,
            linewidth=1,
            color=dmas_col,
            edgecolor='k',
            label='DMAS')

    plt.bar(orr_xs,
            height=[orr_sens[1], orr_spec],
            width=0.75,
            capsize=10,
            linewidth=1,
            color=orr_col,
            edgecolor='k',
            label='ORR')

    plt.legend(fontsize=18,
               loc='upper left',
               framealpha=0.95)

    plt.xticks([1, 4.5],
               ["Sensitivity", "Specificity"],
               size=20)

    plt.ylabel('Metric Value (%)', fontsize=22)

    # Put text on the bars
    das_text_ys = [67, 40]
    das_for_text = [das_sens[1], das_spec]
    dmas_text_ys = [74, 36]
    dmas_for_text = [dmas_sens[1], dmas_spec]
    gd_text_ys = [78, 52]
    gd_for_text = [orr_sens[1], orr_spec]

    for ii in range(len(das_text_ys)):
        plt.text(das_xs[ii], das_text_ys[ii],
                 "%d" % das_for_text[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    for ii in range(len(dmas_text_ys)):
        plt.text(dmas_xs[ii], dmas_text_ys[ii],
                 "%d" % dmas_for_text[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    for ii in range(len(gd_text_ys)):
        plt.text(orr_xs[ii], gd_text_ys[ii],
                 "%d" % gd_for_text[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    # Set appropriate y-limit
    plt.ylim([0, 100])
    plt.tight_layout()  # Make everything fit nicely
    plt.show()  # Display the plot

    # Save fig
    plt.savefig(os.path.join(__OUT_DIR, 'sens_spec_fib_refs.png'),
                dpi=300, transparent=False)


def plt_sens_by_tum_adi():
    """Plot sensitivity as a function of tumour size with adi refs
    """

    # Define x-coords of bars
    das_xs = [0, 3.5, 7, 10.5, 14]
    dmas_xs = [1, 4.5, 8, 11.5, 15]
    orr_xs = [2, 5.5, 9, 12.5, 16]

    # Init fig
    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=20)

    plt.bar(das_xs,
            height=das_by_tum_adi,
            width=0.75,
            linewidth=1,
            color=das_col,
            edgecolor='k',
            label='DAS')
    plt.bar(dmas_xs,
            height=dmas_by_tum_adi,
            width=0.75,
            capsize=10,
            linewidth=1,
            color=dmas_col,
            edgecolor='k',
            label='DMAS')
    plt.bar(orr_xs,
            height=orr_by_tum_adi,
            width=0.75,
            capsize=10,
            linewidth=1,
            color=orr_col,
            edgecolor='k',
            label='ORR')

    plt.legend(fontsize=18,
               loc='upper left',
               framealpha=0.95)

    plt.xticks(dmas_xs,
               ["30", "25", "20", "15", "10"],
               size=20)

    plt.ylabel('Sensitivity (%)', fontsize=22)
    plt.xlabel('Tumour Diameter (mm)', fontsize=22)

    # Put text on the fig
    das_text_ys = np.array(das_by_tum_adi) - 4
    das_text_ys[np.array(das_by_tum_adi) == 0] = 4
    das_for_text = das_by_tum_adi

    dmas_text_ys = np.array(dmas_by_tum_adi) - 4
    dmas_text_ys[np.array(dmas_by_tum_adi) == 0] = 4
    dmas_for_text = dmas_by_tum_adi

    orr_text_ys = np.array(orr_by_tum_adi) - 4
    orr_text_ys[np.array(orr_by_tum_adi) == 0] = 4
    orr_for_text = orr_by_tum_adi

    for ii in range(len(das_text_ys)):
        plt.text(das_xs[ii], das_text_ys[ii],
                 "%d" % das_for_text[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    for ii in range(len(dmas_text_ys)):
        plt.text(dmas_xs[ii], dmas_text_ys[ii],
                 "%d" % dmas_for_text[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    for ii in range(len(orr_text_ys)):
        plt.text(orr_xs[ii], orr_text_ys[ii],
                 "%d" % orr_for_text[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    # Set appropriate y-limit
    plt.ylim([0, 100])
    plt.tight_layout()  # Make everything fit nicely
    plt.show()  # Display the plot

    # Save fig
    plt.savefig(os.path.join(__OUT_DIR, 'sens_by_tum_adi.png'),
                dpi=300, transparent=False)


def plt_sens_by_tum_fib():
    """Plot sensitivity vs tumour size when healthy scan used as refs
    """

    # Define x-coords for bars
    das_xs = [0, 3.5, 7, 10.5, 14]
    dmas_xs = [1, 4.5, 8, 11.5, 15]
    orr_xs = [2, 5.5, 9, 12.5, 16]

    # Init fig
    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=20)

    # Plot bars
    plt.bar(das_xs,
            height=das_by_tum_fib,
            width=0.75,
            linewidth=1,
            color=das_col,
            edgecolor='k',
            label='DAS')
    plt.bar(dmas_xs,
            height=dmas_by_tum_fib,
            width=0.75,
            capsize=10,
            linewidth=1,
            color=dmas_col,
            edgecolor='k',
            label='DMAS')
    plt.bar(orr_xs,
            height=orr_by_tum_fib,
            width=0.75,
            capsize=10,
            linewidth=1,
            color=orr_col,
            edgecolor='k',
            label='ORR')

    plt.legend(fontsize=18,
               loc='upper right',
               framealpha=0.95)

    plt.xticks(dmas_xs,
               ["30", "25", "20", "15", "10"],
               size=20)

    plt.ylabel('Sensitivity (%)', fontsize=22)
    plt.xlabel('Tumour Diameter (mm)', fontsize=22)

    das_text_ys = np.array(das_by_tum_fib) - 4
    das_text_ys[np.array(das_by_tum_fib) == 0] = 4
    das_for_text = das_by_tum_fib

    dmas_text_ys = np.array(dmas_by_tum_fib) - 4
    dmas_text_ys[np.array(dmas_by_tum_fib) == 0] = 4
    dmas_text_ys[np.array(dmas_by_tum_fib) == 5] = 5
    dmas_for_text = dmas_by_tum_fib

    gd_text_ys = np.array(orr_by_tum_fib) - 4
    gd_text_ys[np.array(orr_by_tum_fib) == 0] = 4
    gd_for_text = orr_by_tum_fib

    for ii in range(len(das_text_ys)):
        plt.text(das_xs[ii], das_text_ys[ii],
                 "%d" % das_for_text[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    for ii in range(len(dmas_text_ys)):
        plt.text(dmas_xs[ii], dmas_text_ys[ii],
                 "%d" % dmas_for_text[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    for ii in range(len(gd_text_ys)):
        plt.text(orr_xs[ii], gd_text_ys[ii],
                 "%d" % gd_for_text[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    # Set appropriate y-limit
    plt.ylim([0, 100])
    plt.tight_layout()  # Make everything fit nicely
    plt.show()  # Display the plot

    plt.savefig(os.path.join(__OUT_DIR, 'sens_by_tum_fib.png'),
                dpi=300, transparent=False)


###############################################################################

if __name__ == "__main__":

    # Load the sensitivities for adipose/adipose-fibroglandular
    # reference subtraction
    das_adi_sens, dmas_adi_sens, orr_adi_sens = \
        (load_pickle(
            os.path.join(get_proj_path(), 'output/g3/',
                         'adi_sensitivities_at_target_threshold.pickle')))
    das_fib_sens, dmas_fib_sens, orr_fib_sens = \
        load_pickle(
            os.path.join(get_proj_path(), 'output/g3/',
                         'fib_sensitivities_at_target_threshold.pickle'))

    # Load the specificities
    das_spec, dmas_spec, orr_spec = \
        (load_pickle(
            os.path.join(get_proj_path(), 'output/g3/',
                         'adi_specificities_at_target_threshold.pickle')))

    # Define tuples for plots
    das_sens = das_adi_sens, das_fib_sens
    dmas_sens = dmas_adi_sens, dmas_fib_sens
    orr_sens = orr_adi_sens, orr_fib_sens

    # Load sensitivities as a function of tumour size
    das_by_tum_adi, dmas_by_tum_adi, orr_by_tum_adi = \
        load_pickle(os.path.join(get_proj_path(), 'output/g3/',
                                 'adi_sens_by_tums.pickle'))
    das_by_tum_fib, dmas_by_tum_fib, orr_by_tum_fib = \
        load_pickle(os.path.join(get_proj_path(), 'output/g3/',
                                 'fib_sens_by_tums.pickle'))

    # Plot diagnostic performance when adipose and
    # adipose-fibroglandular scans used as reference
    plt_fib_ref_performance()
    plt_adi_ref_performance()

    # Plot sensitivity as a function of tumour size when adipose
    # scans used as references
    plt_sens_by_tum_adi()
    plt_sens_by_tum_fib()
