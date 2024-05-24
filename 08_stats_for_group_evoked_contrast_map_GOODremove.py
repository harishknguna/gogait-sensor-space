#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

"""
=============================================
Import single subject evoked, take contrast, and do stats
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

def find_closest(arr, val):
    idx = np.abs(arr - val).argmin()
    return arr[idx]


n_subs = len(config_for_gogait.subjects_list)
fs_dir = fetch_fsaverage(verbose=True)
template_subjects_dir = op.dirname(fs_dir)

# ampliNormalization = ['AmpliNorm', 'AmpliActual']
ampliNormalization = 'AmpliActual'

event_type = ['target']
# event_type = ['target']
# version_list = ['GOODremove','CHANremove']
version_list = ['CHANremove']

for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['GOu']
       
      
    contrast_kind = ['NoGo_GOc', 'NoGo_GOu', 'GOu_GOc']    
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
    
    n_subs = len(config_for_gogait.subjects_list)
    num_condi_with_eve = 6
    n_chs = 128 #132
    t_start = - 0.2 # in s
    t_end = 0.7 # in s
    sampling_freq = 500 # in hz
    ncondi = len(condi_name)  # nc = 3
    
    for veri, version in enumerate(version_list):
    
    
        report = mne.Report()
        
        ## n_samples_esti = int(sampling_freq * (isi[ind_isi] + t_end - t_start + 0.002))  
        # estimate the num of time samples per condi/ISI to allocate numpy array
        tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
        tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
        n_samples_esti  = int(500*(tsec_start + tsec_end + 1/500)) # one sample added for zeroth loc
        evoked_data_all_sub_all_condi = np.ones([n_subs, ncondi, n_chs, n_samples_esti])*np.nan
        for sub_num, subject in enumerate(config_for_gogait.subjects_list): 
           
            evoked_data_per_sub_all_condi  = np.ones([ncondi, n_chs, n_samples_esti])*np.nan
            
            for ci, condi in enumerate(condi_name): 
                
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
                 
                data = evoked[0].get_data(picks = 'eeg')
                info = evoked[0].info   # taking info,events,id from the last sub's data (same for all subs)
                data_per_sub = data.copy()
                
                # make sure array size of all the subs/condi are same. sometimes error raised.  
                print(np.shape(data)[1])
                n_samples_actual = np.shape(data)[1] 
                if n_samples_actual == n_samples_esti:
                    evoked_data_per_sub = data_per_sub.copy()
                elif n_samples_actual > n_samples_esti:
                    diff = n_samples_actual - n_samples_esti
                    evoked_data_per_sub = data_per_sub[:,:-diff]
                elif n_samples_actual < n_samples_esti:
                        diff = abs(n_samples_actual - n_samples_esti)
                        dummy_data = np.ones([np.shape(data)[0],diff])*np.nan
                        data_adj = np.hstack((data_per_sub, dummy_data))
                        evoked_data_per_sub = data_adj      
                        
                # store condi in dim1, channels in dim2, time in dim3 
                evoked_data_per_sub_exp_dim = np.expand_dims(evoked_data_per_sub, axis = 0) 
                if ci == 0:
                    evoked_data_per_sub_all_condi = evoked_data_per_sub_exp_dim
                else:
                    evoked_data_per_sub_all_condi = np.vstack(( evoked_data_per_sub_all_condi,  evoked_data_per_sub_exp_dim))  
                          
            # store subs in dim1, condi in dim2, chs in dim3, time in dim4  
            evoked_data_all_sub_all_condi[sub_num,:,:,:] = evoked_data_per_sub_all_condi.copy()
                
                            
        #%% Plot the contrast 
        ## Compute and plot the time averaged contrast maps 
        ## do the contrast per individual and then take average across individuals      
        
        contrast_kind = ['GOu_GOc', 'NoGo_GOu', 'NoGo_GOc']
        time_points_kind = ['T1', 'T2', 'T3', 'T4']
        times  =  np.around(np.arange(-tsec_start,tsec_end,1/sfreq),3)
        report = mne.Report()    
        
        for ti, time_pos in enumerate(time_points_kind): 
            print("for timewindow: %s" % time_pos)       
    
            for ci2, contrast in enumerate(contrast_kind):
                
                if time_pos == 'T1': 
                    tmin = t1min 
                    tmax = t1max 
                    tavg = t1avg
                    tdiff = t1diff
                    timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                    print('plotting T1 activations for ' + contrast)
                elif time_pos == 'T2':
                    tmin = t2min 
                    tmax = t2max 
                    tavg = t2avg
                    tdiff = t2diff
                    timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                    print('plotting T2 activations for ' + contrast)
                elif time_pos == 'T3':
                    tmin = t3min 
                    tmax = t3max 
                    tavg = t3avg
                    tdiff = t3diff
                    timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
                    print('plotting T3 activations for ' + contrast)
                elif time_pos == 'T4':
                    tmin = t4min 
                    tmax = t4max 
                    tavg = t4avg
                    tdiff = t4diff
                    timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
                    print('plotting T4 activations for ' + contrast)
                
                tmin_ind = np.where(times == find_closest(times, tmin))[0][0]
                tmax_ind = np.where(times == find_closest(times, tmax))[0][0]
                
                if contrast == 'GOu_GOc':
                    #before: subs in dim1, condi in dim2, vertices in dim3, time in dim4  
                    #after: subs in dim1, vertices in dim2, time in dim3             
                    X1 = np.mean(evoked_data_all_sub_all_condi[:,1,:,tmin_ind:tmax_ind],axis =2)  # all subs
                    X2 = np.mean(evoked_data_all_sub_all_condi[:,0,:,tmin_ind:tmax_ind],axis =2)  # all subs
                    evoked_contrast_data =  np.mean(X1 - X2, axis = 0) # averaged across subs
                
                elif contrast == 'NoGo_GOu': 
                    X1 = np.mean(evoked_data_all_sub_all_condi[:,2,:,tmin_ind:tmax_ind],axis =2)  # all subs
                    X2 = np.mean(evoked_data_all_sub_all_condi[:,1,:,tmin_ind:tmax_ind],axis =2)  # all subs
                    evoked_contrast_data =  np.mean(X1 - X2, axis = 0) # averaged across subs
                    
                elif contrast == 'NoGo_GOc': 
                    X1 = np.mean(evoked_data_all_sub_all_condi[:,2,:,tmin_ind:tmax_ind],axis =2) # all subs
                    X2 = np.mean(evoked_data_all_sub_all_condi[:,0,:,tmin_ind:tmax_ind],axis =2) # all subs
                    evoked_contrast_data =  np.mean(X1 - X2, axis = 0) # averaged across subs
                
                print("plotting the contrast: %s" % contrast)
                # contrast_evoked = mne.EvokedArray(evoked_contrast_data, info = info, tmin = -0.2, 
                #                        kind='average', baseline = (None,0))
                # Performing the paired sample t-test 
                t_stats, pval = stats.ttest_rel(X1, X2) 
                
                #% PLOTIING the STC MAPS: contrast, t-maps, and p-value
                # plotting the snapshots at 3 different time zones               
                      
                
                
                plt.rcParams.update({'font.size': 20})
                evoked_contrast_data_unit = 1e6 * evoked_contrast_data.copy()
                vmin = np.min(evoked_contrast_data_unit)
                vmax = np.max(evoked_contrast_data_unit)
                vlim = np.max([np.abs(vmin), np.abs(vmax)])
                
                fig0, ax = plt.subplots(1,3, figsize=(15, 5))
                
                # contrast topomap                
                im, cn = mne.viz.plot_topomap(evoked_contrast_data_unit, info, sphere = 0.45, 
                                              axes=ax[0], vlim=(-vlim, vlim))
                ax[0].set_title("{:.3f}".format(tmin) + ' - ' + "{:.3f}".format(tmax) + ' s')
                cax = fig0.colorbar(im, ax=ax[0], shrink=0.35, location = 'left',) 
                cax.set_label(r"$\mu$" + 'V')
                
                # t-values topomap
                vmin = np.min(t_stats)
                vmax = np.max(t_stats)
                vlim = np.max([np.abs(vmin), np.abs(vmax)])
                #fig1, ax = plt.subplots(figsize=(5, 5))
                im, cn = mne.viz.plot_topomap(t_stats, info, sphere = 0.45, 
                                              axes=ax[1], cmap="cool", vlim=(-vlim, vlim))
                ax[1].set_title("{:.3f}".format(tmin) + ' - ' + "{:.3f}".format(tmax) + ' s')
                cax = fig0.colorbar(im, ax=ax[1], shrink=0.35, location = 'left',) 
                cax.set_label("t-values")
                
                # p-values topomap
                plt.rcParams.update({'font.size': 20})
                # create spatial mask
                mask = np.zeros((pval.shape[0], 1), dtype=bool)
                p_accept = 0.05
                ch_inds = np.where(pval < p_accept)[0]
                mask[ch_inds, :] = True
                #fig2, ax = plt.subplots(figsize=(6, 5))
                im, cn = mne.viz.plot_topomap(-pval, info, sphere = 0.45, mask_params=dict(markersize=5),
                                              axes=ax[2], mask=mask, cmap='YlGn', vlim=(-0.05,-0.0001))
                ax[2].set_title("{:.3f}".format(tmin) + ' - ' + "{:.3f}".format(tmax) + ' s')
                cax = fig0.colorbar(im, ax=ax[2], shrink=0.4, location = 'left', 
                                    ticks=[-0.05,-0.04, -0.03, -0.02, -0.01, -0.001])
                cax.ax.set_yticklabels(['0.05','0.04', '0.03', '0.02', '0.01', '0.001'])
                cax.set_label("p-values")
                
                fig0.tight_layout()
                
                report.add_figure(fig0, title = contrast + '_' + timeDur, replace = True)
                
                plt.close('all') 
                
               
               
        # finally saving the report after the for subject loop ends.     
        print('Saving the reports to disk')  
        report.title = 'Stats for group evoked contrast' + ' at ' + evnt + ' across subs (n = ' + str(n_subs) +') ,' + version
        extension = 'Stats_group_evoked_contrast'
        report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
        report.save(report_fname+'_at_'+ evnt +'_' + version+ '.html', overwrite=True)                  
                  
                    
                        
   
   



    