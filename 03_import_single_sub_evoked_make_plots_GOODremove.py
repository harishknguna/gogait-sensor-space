#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

"""
=============================================
import single sub evoked and make Evoked plots and topomaps 

==============================================


"""  



import os.path as op

import mne
from mne.parallel import parallel_func
from mne.channels.montage import get_builtin_montages
from warnings import warn
from pymatreader import read_mat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from scipy import signal
from scipy import stats 
from scipy.linalg import norm
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse 
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid, inset_locator
from mne.stats import spatio_temporal_cluster_test, summarize_clusters_stc

import config_for_gogait
n_subs = len(config_for_gogait.subjects_list)
fs_dir = fetch_fsaverage(verbose=True)
template_subjects_dir = op.dirname(fs_dir)

# ampliNormalization = ['AmpliNorm', 'AmpliActual']
ampliNormalization = 'AmpliActual'

event_type = ['cue']
# event_type = ['cue', 'target']
# version_list = ['GOODremove','CHANremove']
version_list = ['GOODremove']

for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['GOu'] 
    condi_name = ['GOc', 'GOu', 'NoGo']
    time_points_kind = ['T1', 'T2', 'T3', 'T4']
    
    # t1min = 0.100; t1max = 0.130
    # t2min = 0.150; t2max = 0.200
    # t3min = 0.200; t3max = 0.300
    # t4min = 0.350; t4max = 0.450
    
    t1min = 0.075; t1max = 0.125
    t2min = 0.135; t2max = 0.200
    t3min = 0.225; t3max = 0.340
    t4min = 0.360; t4max = 0.470
    
    t1diff = t1max - t1min + 0.002
    t2diff = t2max - t2min + 0.002
    t3diff = t3max - t3min + 0.002
    t4diff = t4max - t4min + 0.002

    t1avg = (t1max + t1min)/2
    t2avg = (t2max + t2min)/2
    t3avg = (t3max + t3min)/2
    t4avg = (t4max + t4min)/2
    
    num_condi_with_eve = 6
    n_chs = 132
    t_start = - 0.2 # in s
    t_end = 0.7 # in s
    sampling_freq = 500 # in hz
    
    for veri, version in enumerate(version_list):
        
        for ci, condi in enumerate(condi_name): 
            report = mne.Report()
            print("condition: %s" % condi)        
       
            ## n_samples_esti = int(sampling_freq * (isi[ind_isi] + t_end - t_start + 0.002))  
            # estimate the num of time samples per condi/ISI to allocate numpy array
            tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
            tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
            n_samples_esti  = int(500*(tsec_start + tsec_end + 1/500)) # one sample added for zeroth loc
            evoked_array_all_sub = np.ones([n_subs, n_chs, n_samples_esti])*np.nan
            for sub_num, subject in enumerate(config_for_gogait.subjects_list): 
                print("Processing subject: %s" % subject)
                eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
                      
                print('  reading the evoked from disk')
                extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_ave'
                evoked_fname = op.join(eeg_subject_dir_GOODremove,
                                         config_for_gogait.base_fname.format(**locals()))
                  
                 # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                print("Input: ", evoked_fname)
               
                if not op.exists(evoked_fname):
                    warn('Run %s not found for subject %s ' %
                           (evoked_fname, subject))
                    continue
                 
                 
                evoked = mne.read_evokeds(evoked_fname)
                 
                evoked_array_per_sub = evoked[0].get_data()
                info = evoked[0].info   # taking info,events,id from the last sub's data (same for all subs)
                
               
                #%% plot the evoked time course and topo maps
               
                              
                evkplt = evoked[0].plot(picks = 'eeg', gfp = True, sphere = 0.4)
               
                report.add_figure(evkplt, title = subject + '_evkplt', replace = True)
                   
                timeavg_evk_topo = evoked[0].plot_topomap(times = [t1avg, t2avg, t3avg, t4avg],
                                   average = [t1diff,t2diff,t3diff,t4diff], sphere = 0.4)
                report.add_figure(timeavg_evk_topo, title = subject + '_evktopo', replace = True)
              
                plt.close('all')
            
            
           
           
            # finally saving the report after the for subject loop ends.     
            print('Saving the reports to disk')  
            report.title = 'Single sub evoked : ' + condi + '_' + evnt+ '_' + version + '_' + ampliNormalization
            #report.title = 'Group sub STC contrast at ' + evnt
            extension = 'single_sub_evoked'
            report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
            report.save(report_fname+'_' + condi +'_'+ evnt +'_' + ampliNormalization + '_' + version+ '.html', overwrite=True)            
                      
   
   



    