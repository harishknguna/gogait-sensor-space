"""
===========
Config file
===========

Configuration parameters for the study.
"""

import os
from collections import defaultdict
import numpy as np
# from pymatreader import read_mat


# ``plot``  : boolean
#   If True, the scripts will generate plots.
#   If running the scripts from a notebook or spyder
#   run %matplotlib qt in the command line to get the plots in extra windows

plot = True
# matplotlib qt

###############################################################################
# DIRECTORIES
# -----------
#
# ``study_path`` : str
#    Set the `study path`` where the data is stored on your system.
#
# Example
# ~~~~~~~
# >>> study_path = '../MNE-sample-data/'
# or
# >>> study_path = '/neurospin/meg/meg_tmp/
#                   Dynacomp_Ciuciu_2011/2019_MEG_Pipeline/'

study_path = '/network/lustre/iss02/cenir/analyse/meeg/GOGAIT/Harish/gogait/'


# ``subjects_dir`` : str
#   The ``subjects_dir`` contains the MRI files for all subjects.

# subjects_dir = os.path.join(study_path, 'subjects')

# ``eeg_dir`` : str
#   The ``eeg_dir`` contains the EEG data in subfolders
#   named my_study_path/EEG/my_subject/

eeg_dir_GOODremove = os.path.join(study_path, 'EEG/040new2_cleantrialdata_500Hz_DC-90Hz_merged_rerefGOOD_remove')
eeg_dir_GOOD = os.path.join(study_path, 'EEG/040new2_cleantrialdata_500Hz_DC-90Hz_merged_rerefGOOD')
eeg_dir_GOODpatients = os.path.join(study_path, 'EEG/140_cleantrialdata_500Hz_1-40Hz_merged_rerefGOOD')

montage_dir = os.path.join(study_path, 'montage')
figure_dir = os.path.join(study_path, 'figures')
report_dir = os.path.join(study_path, 'reports')
report_dir_GOOD = os.path.join(report_dir, 'rerefGOOD')
report_dir_GOODremove = os.path.join(report_dir, 'rerefGOOD_remove')
report_dir_GOODpatients = os.path.join(report_dir, 'rerefGOOD_patients')
ica_report_dir = os.path.join(study_path, 'ica_reports')
cluster_report_dir = os.path.join(study_path, 'cluster_reports_GOODremove')
filtPSD_fig_dir =  os.path.join(figure_dir, 'filtPSD')
events_fig_dir =  os.path.join(figure_dir, 'events')
epochs_fig_dir =  os.path.join(figure_dir, 'epochs')



###############################################################################
# SUBJECTS / RUNS
# ---------------
#
# ``study_name`` : str
#   This is the name of your experiment.
#
# Example
# ~~~~~~~
# >>> study_name = 'MNE-sample'
study_name = 'GOGAIT'

# ``subjects_list`` : list of str
#   To define the list of participants, we use a list with all the anonymized
#   participant names. Even if you plan on analyzing a single participant, it
#   needs to be set up as a list with a single element, as in the 'example'
#   subjects_list = ['SB01']

# To use all subjects use
# subjects_list = ['SB02', 'SB04', 'SB05', 'SB06', 'SB07',
#                  'SB08', 'SB09', 'SB10', 'SB11', 'SB12']

# # # else for speed and fast test you can use:
    
# subjects_list =   ['ATTEM17','BARER19','BARNI44', 'BASGI37', 'BENCH34',
#                     'BIBTH15', 'BONTH22', 'CATFR21', 'CLAPH41',
#                     'DANPA32', 'DASAD26', 'DENLA24', 'DESSY27', 
#                     'FARER31', 'FARYA45', 'FLODO07', 'FONCO13', 
#                     'GRAPH36', 'LATPH43', 'MARFR30', 'PIEMU25', 
#                     'REMCH12', 'SAISA33']

# subjects_list =   ['BARER19','BARNI44', 'BASGI37', 
#                      'BONTH22', 'CATFR21', 'CLAPH41',
#                     'DASAD26', 'DENLA24', 
#                      'FARYA45', 'FLODO07',
#                     'GRAPH36', 'LATPH43', 'PIEMU25', 
#                     'REMCH12', ]

# subjects_list =   ['CATFR21']

# subjects_list = ['GRAPH36', 'LATPH43', 'MARFR30',
#                 'PIEMU25', 'REMCH12', 'SAISA33']

##'HUMMA48' removed
subjects_list_patients =   ['ABSTA46', 'BARGU14', 'BENMO28', 'COUMA29', 'DESJO20',
                            'DEVPA52', 'DROCA16', 'GIRSA40', 'LETER47', 
                            'LOUPH38', 'MAIDO49', 'REMAL39', 'SEGRE51', 'SOUHE53',
                            'SPEAN50' ]

# subjects_list_patients =   ['ABSTA46']

# ``exclude_subjects`` : list of str
#   Now you can specify subjects to exclude from the group study:
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Keep track of the criteria leading you to exclude
# a participant (e.g. too many movements, missing blocks, aborted experiment,
# did not understand the instructions, etc, ...)

# exclude_subjects = ['']
# subjects_list = list(set(subjects_list) - set(exclude_subjects))
# subjects_list.sort()


# ``runs`` : list of str
#   Define the names of your ``runs``
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# The naming should be consistent across participants. List the number of runs
# you ideally expect to have per participant. The scripts will issue a warning
# if there are less runs than is expected. If there is only just one file,
# leave empty!

tasks = ['OFF_GNG'] #['OFF_STATIC', 'OFF_GNG']

trials = ['T1', 'T2', 'T3']  # ['T1'] # ['run01', 'run02']

blocks = ['BLOCK1', 'BLOCK2', 'BLOCK3'] #  ['BLOCK3'] # ['BLOCK3'] #



# ``ch_types``  : list of st
#    The list of channel types to consider.
#
# Example
# ~~~~~~~
# >>> ch_types = ['meg', 'eeg']  # to use MEG and EEG channels
# or
# >>> ch_types = ['meg']  # to use only MEG
# or
# >>> ch_types = ['grad']  # to use only gradiometer MEG channels

ch_types = ['eeg']

# ``base_fname`` : str
#    This automatically generates the name for all files
#    with the variables specified above.
#    Normally you should not have to touch this
base_fname_in_prepro =  '{subject}_All_BLOCKs.mat' # raw brain vision eeg file
# base_fname_in_prepro =  '{subject}_' + '{extension}.mat' # raw brain vision eeg file
# for p atients
base_fname_in_prepro_ON =  '{subject}_ON_BLOCKs.mat' # raw brain vision eeg file
base_fname_in_prepro_OFF =  '{subject}_OFF_BLOCKs.mat' # raw brain vision eeg file

base_fname_in_raw =  study_name + '_{trial}_' + '{subject}_' + '{extension}.vhdr' # raw brain vision eeg file
base_fname_out =  study_name + '_{subject}_' + '{extension}_eeg.fif'   # .vhdr converted into .fif format
base_fname_in_pro =  study_name +  '{subject}_' + '{extension}_eeg.fif'  # down sampled and filtered eeg data
base_fname =  study_name + '_{subject}_' + '{extension}.fif'  
base_fname_npy =  study_name + '_{subject}_' + '{extension}.npy'  # file name of concatenated trials and blocks per subject  
base_fname_no_fif =  study_name + '_{subject}_' + '{extension}'  # file name of concatenated trials and blocks per subject  
base_fname_avg =  study_name + '_' + '{extension}.fif'  # file name of concatenated trials and blocks per subject  
base_fname_generic =  study_name + '_' + '{extension}' 
base_fname_avg_npy =  study_name + '_' + '{extension}.npy'  # file name of concatenated trials and blocks per subject  

base_fname_npy_simple =  '{subject}_' + '{extension}.npy' 
# with study name used to save epochs of data 040new2_cleantrialdata_500Hz_DC-90Hz_merged_rerefGOOD_remove
# base_fname =  '{subject}_' + '{extension}.fif'  # file name of concatenated trials and blocks per subject 
# without study name used to save epochs of data 040new2_cleantrialdata_500Hz_DC-90Hz_merged_rerefGOOD
# base_fname_in_ica =  study_name + '_{trial}_' + '{subject}_' + '{extension}_.fif'


###############################################################################
# BAD CHANNELS
# ------------
# needed for 01-import_and_maxwell_filter.py

# ``bads`` : dict of list | dict of dict
#    Bad channels are noisy sensors that *must* to be listed
#    *before* maxfilter is applied. You can use the dict of list structure
#    of you have bad channels that are the same for all runs.
#    Use the dict(dict) if you have many runs or if noisy sensors are changing
#    across runs.
#
# Example
# ~~~~~~~
# >>> bads = defaultdict(list)
# >>> bads['sample'] = ['MEG 2443', 'EEG 053']  # 2 bads channels
# or
# >>> def default_bads():
# >>>     return dict(run01=[], run02=[])
# >>>
# bads = defaultdict(default_bads)
# bads['subject01'] = dict(run01=['MEG1723', 'MEG1722'], run02=['MEG1723'])
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# During the acquisition of your MEG / EEG data, systematically list and keep
# track of the noisy sensors. Here, put the number of runs you ideally expect
# to have per participant. Use the simple dict if you don't have runs or if
# the same sensors are noisy across all runs.

# For the project GOGAIT, some channels are not compatible with the template/montage of MNE
# consider them as bad channels 
# ValueError: The following electrodes have overlapping positions, which causes problems during visualization:
# O9, O10, FFT9h, FTT9h, FTT10h, FFT10h
# bads = defaultdict(list)
# bads['ATTEM17'] = [ 'F10', 'FT10', 'O9', 'O10', 'FFT9h', 'FTT9h', 'FTT10h', 'FFT10h']

# either put the bad channels here directly
# bads['SB01'] = ['MEG1723', 'MEG1722']
# bads['SB02'] = ['MEG1723', 'MEG1722']
# bads['SB04'] = ['MEG0543', 'MEG2333']
# bads['SB06'] = ['MEG2632', 'MEG2033']

# or read bad channels from textfile in the subject's data folder, named
# bad_channels.txt
#import re
#for subject in subjects_list:
#    bad_chans_file_name = os.path.join(meg_dir,subject,'bad_channels.txt')
#    bad_chans_file = open(bad_chans_file_name,"r") 
#    bad_chans = bad_chans_file.readlines()
#    bad_chans_file.close()
#
#    for i in  bad_chans:            
#        if study_name in i:
#            SBbads = re.findall(r'\d+|\d+.\d+', i)
#    if SBbads:
#        for b, bad in  enumerate(SBbads):
#            SBbads[b] = 'MEG' + str(bad)
#    bads[subject]=SBbads
#    del SBbads
#    
#del subject
 
 
bads_chs = defaultdict(list)
bads_chs['ATTEM17'] = dict(c1 = [], c2 = [])
bads_chs['BARER19'] = dict([])
bads_chs['BARNI44'] = dict([])
bads_chs['BASGI37'] = dict([])
bads_chs['BENCH34'] = dict( c1 = ['O9', 'CP5'],
                            c2 = ['T7'] ,c3 = ['T7', 'POO1', 'O1', 'Oz', 'OI1h'], 
                            c4 = ['T7'], c5 = ['T7'], c6 = ['T7'])
bads_chs['BIBTH15'] = dict(r=[])
bads_chs['BONTH22'] = dict(r=[])
bads_chs['CATFR21'] = dict(r=[])
bads_chs['CLAPH41'] = dict(r=[])
bads_chs['DANPA32'] = dict(r=[])
bads_chs['DASAD26'] = dict(r=[])
bads_chs['DENLA24'] = dict(r=[])
bads_chs['DESSY27'] = dict(r=[])
bads_chs['FARER31'] = dict(r=[])
bads_chs['FARYA45'] = dict(r=[])
bads_chs['FLODO07'] = dict(r=[])
bads_chs['FONCO13'] = dict(r=[])
bads_chs['GRAPH36'] = dict(r=[])
bads_chs['LATPH43'] = dict(r=[])
bads_chs['MARFR30'] = dict(r=[])
bads_chs['PIEMU25'] = dict(r=[])
bads_chs['REMCH12'] = dict(r=[])
bads_chs['SAISA33'] = dict(r=[])

###############################################################################
# BAD TRIALS/ EPOCHS
# ------------
bads_trls = defaultdict(list)
bads_trls['ATTEM17'] = dict(c1 = [], c2 = [])
bads_trls['BARER19'] = dict()
bads_trls['BARNI44'] = dict()
bads_trls['BASGI37'] = dict([])
bads_trls['BENCH34'] = dict()
bads_trls['BIBTH15'] = dict(r=[])
bads_trls['BONTH22'] = dict(r=[])
bads_trls['CATFR21'] = dict(r=[])
bads_trls['CLAPH41'] = dict(r=[])
bads_trls['DANPA32'] = dict(r=[])
bads_trls['DASAD26'] = dict(r=[])
bads_trls['DENLA24'] = dict(r=[])
bads_trls['DESSY27'] = dict(r=[])
bads_trls['FARER31'] = dict(r=[])
bads_trls['FARYA45'] = dict(r=[])
bads_trls['FLODO07'] = dict(r=[])
bads_trls['FONCO13'] = dict(r=[])
bads_trls['GRAPH36'] = dict(r=[])
bads_trls['LATPH43'] = dict(r=[])
bads_trls['MARFR30'] = dict(r=[])
bads_trls['PIEMU25'] = dict(r=[])
bads_trls['REMCH12'] = dict(r=[])
bads_trls['SAISA33'] = dict(r=[])



###############################################################################
# DEFINE ADDITIONAL CHANNELS
# --------------------------
# needed for 01-import_and_maxwell_filter.py

# ``set_channel_types``: dict
#   Here you define types of channels to pick later.
#
# Example
# ~~~~~~~
# >>> set_channel_types = {'EEG061': 'eog', 'EEG062': 'eog',
#                          'EEG063': 'ecg', 'EEG064': 'misc'}

set_channel_types = {'ECG': 'ecg', 'VEOG': 'eog', 'VEOGcalc': 'eog', 'HEOGcalc': 'eog'}


# ``rename_channels`` : dict rename channels
#    Here you name or replace extra channels that were recorded, for instance
#    EOG, ECG.
#
# Example
# ~~~~~~~
# Here rename EEG061 to EOG061, EEG062 to EOG062, EEG063 to ECG063:
# >>> rename_channels = {'EEG061': 'EOG061', 'EEG062': 'EOG062',
#                        'EEG063': 'ECG063'}

rename_channels = None



useTemplateMontage = True


   

# print(easycap_montage)


###############################################################################
# NOTE the order: resampling first and filtering second (ideally the opposite right?) 

# RESAMPLING
# ----------
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# If you have acquired data with a very high sampling frequency (e.g. 2 kHz)
# you will likely want to downsample to lighten up the size of the files you
# are working with (pragmatics)
# If you are interested in typical analysis (up to 120 Hz) you can typically
# resample your data down to 500 Hz without preventing reliable time-frequency
# exploration of your data
#
# ``resample_sfreq``  : float
#   Specifies at which sampling frequency the data should be resampled.
#   If None then no resampling will be done.
#
# Example
# ~~~~~~~
# >>> resample_sfreq = None  # no resampling
# or
# >>> resample_sfreq = 500  # resample to 500Hz

resample_sfreq = 250.  # None

# ``decim`` : int
#   Says how much to decimate data at the epochs level.
#   It is typically an alternative to the `resample_sfreq` parameter that
#   can be used for resampling raw data. 1 means no decimation.
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Decimation requires to lowpass filtered the data to avoid aliasing.
# Note that using decimation is much faster than resampling.
#
# Example
# ~~~~~~~
# >>> decim = 1  # no decimation
# or
# >>> decim = 4  # decimate by 4 ie devide sampling frequency by 4

decim = 1

###############################################################################
# FREQUENCY FILTERING
# -------------------
# done in 01-import_and_maxwell_filter.py

# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# It is typically better to set your filtering properties on the raw data so
# as to avoid what we call border (or edge) effects.
#
# If you use this pipeline for evoked responses, you could consider
# a low-pass filter cut-off of h_freq = 40 Hz
# and possibly a high-pass filter cut-off of l_freq = 1 Hz
# so you would preserve only the power in the 1Hz to 40 Hz band.
# Note that highpass filtering is not necessarily recommended as it can
# distort waveforms of evoked components, or simply wash out any low
# frequency that can may contain brain signal. It can also act as
# a replacement for baseline correction in Epochs. See below.
#
# If you use this pipeline for time-frequency analysis, a default filtering
# could be a high-pass filter cut-off of l_freq = 1 Hz
# a low-pass filter cut-off of h_freq = 120 Hz
# so you would preserve only the power in the 1Hz to 120 Hz band.
#
# If you need more fancy analysis, you are already likely past this kind
# of tips! :)


# ``l_freq`` : float
#   The low-frequency cut-off in the highpass filtering step.
#   Keep it None if no highpass filtering should be applied.

l_freq = 1.

# ``h_freq`` : float
#   The high-frequency cut-off in the lowpass filtering step.
#   Keep it None if no lowpass filtering should be applied.

# ValueError: lowpass frequency 125.0 must be less than Nyquist (125.0)
# keep some margin
margin = 10 #Hz
h_freq = resample_sfreq/2 - margin

# NOTCJ filtering to remove power line noise at 50 Hz
p_freq = 50 




###############################################################################
# AUTOMATIC REJECTION OF ARTIFACTS
# --------------------------------
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Have a look at your raw data and train yourself to detect a blink, a heart
# beat and an eye movement.
# You can do a quick average of blink data and check what the amplitude looks
# like.
#
#  ``reject`` : dict | None
#    The rejection limits to make some epochs as bads.
#    This allows to remove strong transient artifacts.
#    If you want to reject and retrieve blinks later, e.g. with ICA,
#    don't specify a value for the eog channel (see examples below).
#    Make sure to include values for eeg if you have EEG data
#
# Note
# ~~~~
# These numbers tend to vary between subjects.. You might want to consider
# using the autoreject method by Jas et al. 2018.
# See https://autoreject.github.io
#
# Example
# ~~~~~~~
# >>> reject = {'grad': 4000e-13, 'mag': 4e-12, 'eog': 150e-6}
# >>> reject = {'grad': 4000e-13, 'mag': 4e-12, 'eeg': 200e-6}
# >>> reject = None

reject = {'eeg': 45e-6}

###############################################################################
# EPOCHING
# --------
#
# ``tmin``: float
#    A float in seconds that gives the start time before event of an epoch.
#
# Example
# ~~~~~~~
# >>> tmin = -0.2  # take 200ms before event onset.

tmin = -0.5

# ``tmax``: float
#    A float in seconds that gives the end time before event of an epoch.
#
# Example
# ~~~~~~~
# >>> tmax = 0.5  # take 500ms after event onset.

tmax = 1.5

# ``trigger_time_shift`` : float | None
#    If float it specifies the offset for the trigger and the stimulus
#    (in seconds). You need to measure this value for your specific
#    experiment/setup.
#
# Example
# ~~~~~~~
# >>> trigger_time_shift = 0  # don't apply any offset

#  trigger_time_shift = -0.0416

# ``baseline`` : tuple
#    It specifies how to baseline the epochs; if None, no baseline is applied.
#
# Example
# ~~~~~~~
# >>> baseline = (None, 0)  # baseline between tmin and 0

# There is an event 500ms prior to the time-locking event, so we want
# to take a baseline before that
baseline = (None, 0.)

# ``stim_channel`` : str
#    The name of the stimulus channel, which contains the events.
#
# Example
# ~~~~~~~
# >>> stim_channel = 'STI 014'  # or 'STI101'

stim_channel = 'STI101'

# ``min_event_duration`` : float
#    The minimal duration of the events you want to extract (in seconds).
#
# Example
# ~~~~~~~
# >>> min_event_duration = 0.002  # 2 miliseconds

min_event_duration = 0.002

#  `event_id`` : dict
#    Dictionary that maps events (trigger/marker values)
#    to conditions.
#
# Example
# ~~~~~~~
# >>> event_id = {'auditory/left': 1, 'auditory/right': 2}`
# or
# >>> event_id = {'Onset': 4} with conditions = ['Onset']

event_id = {'start':1, 'cue/green': 2, 'cue/red': 4,
            'target/green': 8, 'target/red': 16}

#  `conditions`` : dict
#    List of condition names to consider. Must match the keys
#    in event_id.
#
# Example
# ~~~~~~~
# >>> conditions = ['auditory', 'visual']
# or
# >>> conditions = ['left', 'right']

# conditions = ['start','cue', 'target']
conditions = ['cue/green', 'cue/red', 'target/green', 'target/red' ]

###############################################################################
# ARTIFACT REMOVAL
# ----------------
#
# You can choose between ICA and SSP to remove eye and heart artifacts.
# SSP: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ssp.html?highlight=ssp # noqa
# ICA: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ica.html?highlight=ica # noqa
# if you choose ICA, run scripts 5a and 6a
# if you choose SSP, run scripts 5b and 6b
#
# Currently you cannot use both.
#
# ``use_ssp`` : bool
#    If True ICA should be used or not.

use_ssp = False

# ``use_ica`` : bool
#    If True ICA should be used or not.

use_ica = True

# ``ica_decim`` : int
#    The decimation parameter to compute ICA. If 5 it means
#    that 1 every 5 sample is used by ICA solver. The higher the faster
#    it is to run but the less data you have to compute a good ICA.

ica_decim = 4


# ``default_reject_comps`` : dict
#    A dictionary that specifies the indices of the ICA components to reject
#    for each subject. For example you can use:
#    rejcomps_man['subject01'] = dict(eeg=[12], meg=[7])

def default_reject_comps():
    return dict(meg=[], eeg=[])


rejcomps_man = defaultdict(default_reject_comps)


# ``ica_ctps_ecg_threshold``: float
#    The threshold parameter passed to `find_bads_ecg` method.

ica_ctps_ecg_threshold = 0.1

# ``ica_correlation_eog_threshold``: float
#    The threshold parameter passed to `find_bads_eog` method.

ica_eog_threshold = 'auto' #3.

###############################################################################
# DECODING
# --------
#
# ``decoding_conditions`` : list
#    List of conditions to be classified.
#
# Example
# ~~~~~~~
# >>> decoding_conditions = []  # don't do decoding
# or
# >>> decoding_conditions = [('auditory', 'visual'), ('left', 'right')]

decoding_conditions = [('incoherent', 'coherent')]

# ``decoding_metric`` : str
#    The metric to use for cross-validation. It can be 'roc_auc' or 'accuracy'
#    or any metric supported by scikit-learn.

decoding_metric = 'roc_auc'

# ``decoding_n_splits`` : int
#    The number of folds (a.k.a. splits) to use in the cross-validation.

decoding_n_splits = 5

###############################################################################
# TIME-FREQUENCY
# --------------
#
# ``time_frequency_conditions`` : list
#    The conditions to compute time-frequency decomposition on.

time_frequency_conditions = ['coherent']

###############################################################################
# SOURCE SPACE PARAMETERS
# -----------------------
#

# ``spacing`` : str
#    The spacing to use. Can be ``'ico#'`` for a recursively subdivided
#    icosahedron, ``'oct#'`` for a recursively subdivided octahedron,
#    ``'all'`` for all points, or an integer to use appoximate
#    distance-based spacing (in mm).

spacing = 'oct6'

# ``mindist`` : float
#    Exclude points closer than this distance (mm) to the bounding surface.

mindist = 5

# ``loose`` : float in [0, 1] | 'auto'
#    Value that weights the source variances of the dipole components
#    that are parallel (tangential) to the cortical surface. If loose
#    is 0 then the solution is computed with fixed orientation,
#    and fixed must be True or "auto".
#    If loose is 1, it corresponds to free orientations.
#    The default value ('auto') is set to 0.2 for surface-oriented source
#    space and set to 1.0 for volumetric, discrete, or mixed source spaces,
#    unless ``fixed is True`` in which case the value 0. is used.

loose = 0.2

# ``depth`` : None | float | dict
#    If float (default 0.8), it acts as the depth weighting exponent (``exp``)
#    to use (must be between 0 and 1). None is equivalent to 0, meaning no
#    depth weighting is performed. Can also be a `dict` containing additional
#    keyword arguments to pass to :func:`mne.forward.compute_depth_prior`
#    (see docstring for details and defaults).

depth = 0.8

# method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
#    Use minimum norm, dSPM (default), sLORETA, or eLORETA.

method = 'dSPM'

# smooth : int | None
#    Number of iterations for the smoothing of the surface data.
#    If None, smooth is automatically defined to fill the surface
#    with non-zero values. The default is spacing=None.

smooth = 10


# ``base_fname_trans`` : str
#   The path to the trans files obtained with coregistration.
#
# Example
# ~~~~~~~
# >>> base_fname_trans = '{subject}_' + study_name + '_raw-trans.fif'
# or
# >>> base_fname_trans = '{subject}-trans.fif'

base_fname_trans = '{subject}-trans.fif'

# XXX not needed
# fsaverage_vertices = [np.arange(10242), np.arange(10242)]

if not os.path.isdir(study_path):
    os.mkdir(study_path)

# if not os.path.isdir(subjects_dir):
#     os.mkdir(subjects_dir)

###############################################################################
# ADVANCED
# --------
#
# ``l_trans_bandwidth`` : float | 'auto'
#    A float that specifies the transition bandwidth of the
#    highpass filter. By default it's `'auto'` and uses default mne
#    parameters.

l_trans_bandwidth = 'auto'

#  ``h_trans_bandwidth`` : float | 'auto'
#    A float that specifies the transition bandwidth of the
#    lowpass filter. By default it's `'auto'` and uses default mne
#    parameters.

h_trans_bandwidth = 'auto'

#  ``N_JOBS`` : int
#    An integer that specifies how many subjects you want to run in parallel.

N_JOBS = 4

# ``random_state`` : None | int | np.random.RandomState
#    To specify the random generator state. This allows to have
#    the results more reproducible between machines and systems.
#    Some methods like ICA need random values for initialisation.

random_state = 42

# ``shortest_event`` : int
#    Minimum number of samples an event must last. If the
#    duration is less than this an exception will be raised.

shortest_event = 1

# ``allow_maxshield``  : bool
#    To import data that was recorded with Maxshield on before running
#    maxfilter set this to True.

allow_maxshield = True

###############################################################################
# CHECKS
# --------
#
# --- --- You should not touch the next lines --- ---

# if (use_maxwell_filter and
#         len(set(ch_types).intersection(('meg', 'grad', 'mag'))) == 0):
#     raise ValueError('Cannot use maxwell filter without MEG channels.')

# if use_ssp and use_ica:
#     raise ValueError('Cannot use both SSP and ICA.')
