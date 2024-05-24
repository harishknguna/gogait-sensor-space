#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================
07. Importing fieldtrip pre-processed eeg data
==============================================
Imports the data that has been pre-processed in fieldtrip (ft) and converts into
MNE compatable epochs structure

"""  

import os.path as op

import mne
from mne.parallel import parallel_func
from mne.channels.montage import get_builtin_montages
from warnings import warn
# from pymatreader import read_mat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import config_for_gogait

# subject = config_for_gogait.subjects_list[0]

version_list = ['GOODremove','CHANremove']
ep_extension = 'TF'

 
t1min = 0.100; t1max = 0.130
t2min = 0.150; t2max = 0.200
t3min = 0.200; t3max = 0.300
t4min = 0.350; t4max = 0.450

t1diff = t1max - t1min + 0.002
t2diff = t2max - t2min + 0.002
t3diff = t3max - t3min + 0.002
t4diff = t4max - t4min + 0.002

t1avg = (t1max + t1min)/2
t2avg = (t2max + t2min)/2
t3avg = (t3max + t3min)/2
t4avg = (t4max + t4min)/2
                  

for subject in config_for_gogait.subjects_list:
    for veri, version in enumerate(version_list):
    
        eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
        print("Processing subject: %s" % subject)
        
        report = mne.Report(title = subject + '_' + version)
        
    #%%         
       
        """ step 1: import numpy array epochs of different conditions and ISI"""    
        
        event_type = ['cue','target']
        condi_name = ['GOc', 'GOu', 'NoGo']

        for ei, evnt in enumerate(event_type):
            for ci, condi in enumerate(condi_name):    
                  print('  Importing the sub-epochs from disk')
                  
                  ## added on 15/01/2024 (for HC) and 22/05/2024 (for PD)
                  ## importing epochs separately for evoked and TF with different prestim lengths 
                  if ep_extension == 'TF':
                      extension =  condi_name[ci] +'_' + event_type[ei] + '_eeg_prep_'+ ep_extension+'.npy'
                  else:
                      extension =  condi_name[ci] +'_' + event_type[ei] + '_eeg_prep.npy'
                      
                  npy_array_fname = op.join(eeg_subject_dir_GOODremove,
                                      config_for_gogait.base_fname_no_fif.format(**locals()))   
                  
                  print("Input: ", npy_array_fname)
                  print("Output: None")       
              
                  if not op.exists(npy_array_fname):
                      warn('Run %s not found for subject %s ' %
                            (npy_array_fname, subject))
                      continue
                
                
                  data = np.load(npy_array_fname)   
                
               
                  ## importing the info from ...info.fif file
                  extension = 'info'
                  info_fname = op.join(eeg_subject_dir_GOODremove,
                                        config_for_gogait.base_fname.format(**locals()))
               
                  info = mne.io.read_info(info_fname)
                  
                  if version == 'CHANremove': # if needed, remove the bads from epochs
                      info['bads'] = ['Fp1', 'Fp2', 'AFp1', 'AFp2', 'AF7', 'AF8', 'F9', 'F10', 
                                             'FT9', 'FT10', 'P9', 'O9', 'PO9', 'O10', 'PO10', 'P10', 'TP9', 'TP10',
                                             'TPP9h', 'PPO9h', 'OI1h', 'OI2h', 'POO10h', 'PPO10h', 'TPP10h'] # 'Iz',
                  elif version == 'GOODremove':
                      info['bads'] = []
                      
                  
                  sampling_freq = info['sfreq']
        
                  num_events = np.shape(data)[0]
                  epoch_len_in_samples = np.shape(data)[2] 
                  events = np.column_stack(
                  (
                      np.arange(0, num_events*epoch_len_in_samples, epoch_len_in_samples),
                      np.zeros(num_events, dtype=int),
                      np.zeros(num_events, dtype=int))) # marking events at zero
                      # labeling S2 GOc event id as '1'; S4 GOu as '2'; S4 NoGo as '3'
                  if evnt == 'cue' and condi == 'GOc':
                     events[:, 2] = 1*np.ones(num_events, dtype=int)
                     event_dict = dict(S2=1)
                  elif evnt == 'cue' and condi == 'GOu':
                     events[:, 2] = 2*np.ones(num_events, dtype=int)
                     event_dict = dict(S4=2)
                  elif evnt == 'cue' and condi == 'NoGo':
                     events[:, 2] = 3*np.ones(num_events, dtype=int)
                     event_dict = dict(S4=3)
                  elif evnt == 'target' and condi == 'GOc':
                     events[:, 2] = 4*np.ones(num_events, dtype=int)
                     event_dict = dict(S8=4)
                  elif evnt == 'target' and condi == 'GOu':
                     events[:, 2] = 5*np.ones(num_events, dtype=int)
                     event_dict = dict(S8=5)
                  elif evnt == 'target' and condi == 'NoGo':
                     events[:, 2] = 6*np.ones(num_events, dtype=int)
                     event_dict = dict(S16=6)
                 
                  ## added on 15/01/2024
                  ## saving epochs separately for evoked and TF with different prestim lengths 
                  if evnt == 'cue' and ep_extension == 'TF':
                      tsec_start = 0.8 # pre-stimulus(t2/S2/S4) duration in sec
                      tsec_end = 1.3 # post-stimulus (t3/S8/S16) duration in sec
                  elif evnt == 'target' and ep_extension == 'TF':
                      tsec_start = 1.0 # pre-stimulus(t2/S2/S4) duration in sec
                      tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
                  else:
                      tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
                      tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec      
            
                  ## BASELINE CORRECTION APPLIED ## to be corrected for HC for TF?? 22/05
                  ## should bslncorr applied twice during epochs and TF stages?? check with NG.  
                  epochs = mne.EpochsArray(data, info = info, tmin = - tsec_start,  
                                          events = events, event_id = event_dict, baseline = (None,0))
                  epochs.set_channel_types(config_for_gogait.set_channel_types)
                  # epochs.plot(picks = ['all'], events=events, event_id=event_dict)
                  evoked = epochs.average(picks = ['all'])
                  # %%   saving the epoch data in numpy array
                  print('  Writing the sub-epochs numpy array to disk')
                  extension = condi_name[ci] +'_' + event_type[ei] +'_' + version +'_' + ep_extension + '_epo'
                  epochs_fname = op.join(eeg_subject_dir_GOODremove,
                                            config_for_gogait.base_fname.format(**locals()))
                 
                  # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                  print("Output: ", epochs_fname)
                  epochs.save(epochs_fname, overwrite=True)     
                
                  # #%%   saving the evoked data in numpy array
                  # print('  Writing the evoked to disk')
                  # extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_ave'
                  # evoked_fname = op.join(eeg_subject_dir_GOODremove,
                  #                         config_for_gogait.base_fname.format(**locals()))
                 
                  # # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                  # print("Output: ", evoked_fname)
                  # evoked.save(evoked_fname, overwrite=True)     
                
                
       
    
    
   
        

