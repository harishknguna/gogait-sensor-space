#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

"""
=============================================
15. Single sub analysis: Source reconstruction using template MRI
https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html#sphx-glr-auto-tutorials-forward-35-eeg-no-mri-py
https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html

==============================================
Imports the data that has been pre-processed in fieldtrip (ft) and converts into
MNE compatable epochs structure

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

event_type = ['target']
# event_type = ['target']
# version_list = ['GOODremove','CHANremove']
version_list = ['CHANremove']

# t1min = 0.075; t1max = 0.125
# t2min = 0.135; t2max = 0.200
# t3min = 0.280; t3max = 0.340
# t4min = 0.360; t4max = 0.470

# tmin = [0.075, 0.135, 0.225, 0.360]
# tmax = [0.125, 0.200, 0.340, 0.470]

# reversing the time windows highest to lowest, 
# in reference to the channel order frontal to occipital:
tmin = [0.360, 0.225, 0.135, 0.075]
tmax = [0.470, 0.340, 0.200, 0.125]

tdiff  = []

for ti in range(4):
    diff = round(tmax[ti] - tmin[ti] + 0.002, 3)
    tdiff.append(diff)
    
tavg  = []

for ti in range(4):
    avg = round((tmax[ti] + tmin[ti])/2, 3)
    tavg.append(avg)               
               
for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['GOu']
       
    norm_kind = ['vector','normVec', 'normVec_zsc']  
    norm_kind = norm_kind[2] # selecting the norm_zscored option
    
    contrast_kind = ['NoGo_GOc', 'NoGo_GOu', 'GOu_GOc']    
    condi_name = ['GOc', 'GOu', 'NoGo']
    time_points_kind = ['T1', 'T2', 'T3', 'T4']    
    num_condi_with_eve = 6
    n_chs = 132
    t_start = - 0.2 # in s
    t_end = 0.7 # in s
    sampling_freq = 500 # in hz
    
    for veri, version in enumerate(version_list):
    
    
        report = mne.Report()
        
        for ci, condi in enumerate(condi_name): 
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
                 
                data = evoked[0].get_data()
                #data = data.reshape(1,np.shape(data)[0],np.shape(data)[1])
                print(np.shape(data)[1])
                n_samples_actual = np.shape(data)[1] 
                if n_samples_actual == n_samples_esti:
                    evoked_array_all_sub[sub_num,:,:] = data
                elif n_samples_actual > n_samples_esti:
                    diff = n_samples_actual - n_samples_esti
                    evoked_array_all_sub[sub_num,:,:] = data[:,:-diff]
                elif n_samples_actual < n_samples_esti:
                        diff = abs(n_samples_actual - n_samples_esti)
                        dummy_data = np.ones([np.shape(data)[0],diff])*np.nan
                        data_adj = np.hstack((data, dummy_data))
                        evoked_array_all_sub[sub_num,:,:] = data_adj      
            
            # store condi in dim1, subs in dim2, chs in dim3, time in dim4  
            evoked_array_all_sub_exp_dim = np.expand_dims(evoked_array_all_sub, axis = 0)
                
                            
            if ci == 0:
                evoked_array_all_sub_all_condi = evoked_array_all_sub_exp_dim
            else:
                evoked_array_all_sub_all_condi = np.vstack(( evoked_array_all_sub_all_condi, evoked_array_all_sub_exp_dim))
            
            avg_evoked_all_condi = np.mean(evoked_array_all_sub_all_condi, axis = 1) 
                
        #%% # do the contrast per individual and then take average across individuals
        ## contrast taken in zscored scale: 
        ## condi_name = ['GOc', 'GOu', 'NoGo'] # check the indices [0, 1, 2] 
        
               
        info = evoked[0].info   # taking info,events,id from the last sub's data (same for all subs)
   
        evoked_GOc = mne.EvokedArray(avg_evoked_all_condi[0,:,:], info = info, tmin = -0.2, 
                               kind='average', baseline = (None,0))
        evoked_GOu = mne.EvokedArray(avg_evoked_all_condi[1,:,:], info = info, tmin = -0.2, 
                               kind='average', baseline = (None,0))
        evoked_NoGo = mne.EvokedArray(avg_evoked_all_condi[2,:,:], info = info, tmin = -0.2, 
                               kind='average', baseline = (None,0))
        
        # evoked_NoGo.plot_joint(picks = 'eeg')
        
        # report.add_figure(evkplt, title = contrast +'_evkplt', replace = pTrue)
        #CH = ['FC1', 'Cz', 'P1', 'PO9']
        # CH = ['Oz', 'Pz', 'Cz', 'Fz']
        CH = ['Fz', 'Cz', 'Pz', 'Oz']
        # ch_inds = info['ch_names'].index(('FC1','Cz'))  
        ch_inds_all = [i for i in range(len(info['ch_names'])) if info['ch_names'][i] in CH]
        
        for ch, ch_ind in enumerate(ch_inds_all):
            data0 = avg_evoked_all_condi[0,ch_ind,:].reshape(-1,1)
            data0 = 1e6 * data0 ## units conversion 
            data1 = avg_evoked_all_condi[1,ch_ind,:].reshape(-1,1)
            data1 = 1e6 * data1 ## units conversion 
            data2 = avg_evoked_all_condi[2,ch_ind,:].reshape(-1,1)
            data2 = 1e6 * data2 ## units conversion 
            data = np.hstack((data0, data1, data2))
            times = evoked[0].times
            
        
            fig1, ax1 = plt.subplots(figsize=(10,6))
            ax1.plot(times, data0, linewidth = 3, color = 'g', label = 'GOc')
            ax1.plot(times, data1, linewidth = 3, color = 'b', label = 'GOu')
            ax1.plot(times, data2, linewidth = 3, color = 'r', label = 'NoGo')
            ax1.set_ylabel('Amplitude (' + r"$\mu$" + 'V)' , fontsize = '20')
            ax1.axvline(x = 0, color = 'k', linestyle = '--') 
            ax1.set_xlabel('Time (s)', fontsize = '20')
            ax1.set_title(CH[ch], fontsize = '20')
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.ylim([-7, 4])
            
            if CH[ch] == 'Fz':
                plt.rcParams.update({'font.size': 20})
                plt.legend(loc='lower left')
            
            ax1.axvspan(tmin[ch], tmax[ch], facecolor='k', alpha = 0.1)                
            ax1.grid(True)
            plt.rcParams.update({'font.size': 20})
            fig1.tight_layout()
            report.add_figure(fig1, title = 'evoked_' + CH[ch], replace = True)
        
            # for evoked mask = (ch, times)
            mask = np.zeros((128, 450), dtype=bool)
            mask[ch_inds_all[ch],:] = True
            
            # plotting the snapshots at 3 different time zones
            plt.rcParams.update({'font.size': 10})
         
            im = evoked_NoGo.plot_topomap(times = tavg[ch], average = tdiff[ch], 
                                          sphere = 0.45, mask=mask, 
                                          mask_params = dict(marker='o', markerfacecolor='w',
                                                             markeredgecolor='k', linewidth=0,
                                                             markersize=15))
            
            report.add_figure(im, title = 'topo_' + CH[ch], replace = True)
      
        plt.close('all') 
                
               
               
        # finally saving the report after the for subject loop ends.     
        print('Saving the reports to disk')  
        report.title = 'Group evoked classical plots at_' + evnt+ '_across subjects (n = ' + str(n_subs) + ') ,' + version
        #report.title = 'Group sub STC contrast at ' + evnt
        extension = 'group_evoked_classical_plots'
        report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
        report.save(report_fname+'_at_'+ evnt +'_'+ version+ '.html', overwrite=True)            
          
   
   



    