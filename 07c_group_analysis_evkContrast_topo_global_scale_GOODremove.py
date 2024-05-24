#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================
09. Group analysis: averaged ERP, topomaps
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
ampliNormalization = 'AmpliActual'

# t1min = 0.075; t1max = 0.125
# t2min = 0.135; t2max = 0.200
# t3min = 0.280; t3max = 0.340
# t4min = 0.360; t4max = 0.470

## GFP based timings
# tmin = [0.075, 0.135, 0.225, 0.360]
# tmax = [0.125, 0.200, 0.340, 0.470]

# ## Ziri et al 2024 (in press)
# tmin = [0.100, 0.150, 0.200, 0.250, 0.350]
# tmax = [0.130, 0.200, 0.300, 0.350, 0.450]

# for ppt on 08/03/2024
# t1min = 0.136; t1max = 0.164
tmin = [0.160, 0.300, 0.370] 
tmax = [0.180, 0.350, 0.450]

# reversing the time windows highest to lowest, 
# in reference to the channel order frontal to occipital:
# tmin = [0.360, 0.225, 0.135, 0.075]
# tmax = [0.470, 0.340, 0.200, 0.125]

nT = len(tmin)
tdiff  = []

for ti in range(nT):
    diff = round(tmax[ti] - tmin[ti] + 0.002, 3)
    tdiff.append(diff)
    
tavg  = []

for ti in range(nT):
    avg = round((tmax[ti] + tmin[ti])/2, 3)
    tavg.append(avg)               
               
for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
   
       
    norm_kind = ['vector','normVec', 'normVec_zsc']  
    norm_kind = norm_kind[2] # selecting the norm_zscored option
    
    contrast_kind = ['GOu_GOc','NoGo_GOu']      
   
    ncondi = len(contrast_kind)
    # time_points_kind = ['T1', 'T2', 'T3', 'T4', 'T5'] 
    time_points_kind = ['T1', 'T2', 'T3']  
    num_condi_with_eve = 6
    n_chs = 132
    t_start = - 0.2 # in s
    t_end = 0.7 # in s
    sampling_freq = 500 # in hz
    
    for veri, version in enumerate(version_list):
    
    
        report = mne.Report()
        
        for ci, condi in enumerate(contrast_kind): 
            
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
                # extension =  contrast_kind[ci] +'_' + event_type[ei] +'_' + version +'_ave'
                extension =  condi +'_' + evnt +'_' + version +'_' + ampliNormalization +'_ave'
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
            
     
        
#%% Plotting the topomaps
    
info = evoked[0].info   # taking info,events,id from the last sub's data (same for all subs)

evoked_GOu_GOc = mne.EvokedArray(avg_evoked_all_condi[0,:,:], info = info, tmin = -0.2, 
                       kind='average', baseline = (None,0))
evoked_NoGo_GOu = mne.EvokedArray(avg_evoked_all_condi[1,:,:], info = info, tmin = -0.2, 
                       kind='average', baseline = (None,0))

 
#%% find global min/max
vabsmax_all = np.ones((1,nT))*np.nan
for i in range(nT): 
    time_avg_GOu_GOc = evoked_GOu_GOc.get_data(picks = 'eeg', tmin = tmin[i], tmax = tmax[i]).mean(axis = 1)
    time_avg_NoGo_GOu = evoked_NoGo_GOu.get_data(picks = 'eeg',tmin = tmin[i], tmax = tmax[i]).mean(axis = 1)
    vabsmax_all[:,i] = np.max([np.max(np.abs(time_avg_GOu_GOc)), np.max(np.abs(time_avg_NoGo_GOu))])

vabsmax = 1e6*np.max(vabsmax_all)
vmaxGlobal = + vabsmax
vminGlobal = - vabsmax

 
#%% 

## don't forget to run, %matplotlib qt, else topos wont be plotted
scale = ['timeScale', 'globalScale']
for fiscl in scale: 
    fig, axs = plt.subplots(ncondi,nT, figsize=(9,2.8), sharex=True, sharey=True)
    for cols in range(nT):
        for rows in range(ncondi): 
            time_avg_GOu_GOc = evoked_GOu_GOc.get_data(picks = 'eeg', tmin = tmin[i], tmax = tmax[i]).mean(axis = 1)
            time_avg_NoGo_GOu = evoked_NoGo_GOu.get_data(picks = 'eeg',tmin = tmin[i], tmax = tmax[i]).mean(axis = 1)
           
            ## for column wise scale, uncomment these lines
            vabsmax = 1e6*np.max([np.max(np.abs(time_avg_GOu_GOc)),np.max(np.abs(time_avg_NoGo_GOu))])
            vmaxTime = + vabsmax
            vminTime = - vabsmax
            if fiscl == 'timeScale':
                vmin = vminTime
                vmax = vmaxTime
            elif fiscl == 'globalScale':
                vmin = vminGlobal
                vmax = vmaxGlobal
            
            
            if rows == 0:
                im = evoked_GOu_GOc.plot_topomap(times = tavg[cols], average = tdiff[cols], vlim=(vmin, vmax),
                                              sphere = 0.45, axes = axs[rows, cols], colorbar=False)
                if cols == 0:
                    axs[rows, cols].set_ylabel("GOu_GOc")
            elif rows == 1:
                im = evoked_NoGo_GOu.plot_topomap(times = tavg[cols], average = tdiff[cols],  vlim=(vmin, vmax),
                                              sphere = 0.45, axes = axs[rows, cols], colorbar=False,)
                if cols == 0:
                    axs[rows, cols].set_ylabel("NoGo_GOu")
                # remove the title 
                axs[rows, cols].set_title("")
                
                ## colorbar of topo
                if fiscl == 'timeScale' or cols == nT-1:
                    ## create additional axes (for ERF and colorbar)
                    divider = make_axes_locatable(axs[rows, cols])
                    # add axes for colorbar
                    ax_colorbar = divider.append_axes("right", size="5%", pad=0.01)
                    image = axs[rows, cols].images[0]
                    plt.colorbar(image, cax=ax_colorbar) # label =  r"$\mu$" + 'V')
                    ax_colorbar.set_title(r"$\mu$" + 'V', fontsize = '8')
                
    
    fig.tight_layout()
    report.add_figure(fig, title = fiscl, replace = True)
    plt.close('all') 

#%% finally saving the report after the for condi loop ends.     
print('Saving the reports to disk')  
report.title = 'Group (n = ' + str(n_subs) + ') evkContrast topo_'+ evnt + ': ' + version 
extension = 'group_evkContrast_topo'
report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
report.save(report_fname+'_at_'+ evnt + '_' + version + '_ppt.html', overwrite=True)                  
            
     
     
         
  
   
        

