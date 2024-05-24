#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:12:46 2023

@author: harish.gunasekaran
"""

"""
=============================================
Plot GFPs 
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

import config_for_gogait



# version_list = ['GOODremove','CHANremove']
version_list = ['GOODremove']
n_subs = len(config_for_gogait.subjects_list)

""" step 1: import epochs of different conditions and ISI"""    


# t1min = 0.100; t1max = 0.130
# t2min = 0.150; t2max = 0.200
# t3min = 0.200; t3max = 0.300
# t4min = 0.350; t4max = 0.450


# t1min = 0.075; t1max = 0.125
# t2min = 0.135; t2max = 0.200
# t3min = 0.225; t3max = 0.340
# t4min = 0.360; t4max = 0.470

 # for ppt on 08/03/2024
 # t1min = 0.136; t1max = 0.164
t1min = 0.160; t1max = 0.180
t2min = 0.300; t2max = 0.350
t3min = 0.370; t3max = 0.450

event_type = ['cue', 'target']
condi_name = ['GOc', 'GOu', 'NoGo']

num_condi_with_eve = 6
n_chs = 132
t_start = - 0.2 # in s
t_end = 0.7 # in s
sampling_freq = 500 # in hz
for veri, version in enumerate(version_list):
    # create mne reports for saving plots 
    report = mne.Report(title = 'Group analysis: compare evokeds (n = 23), ' + version ) 
    #% for loop of conditions with ISIs
    
    evoked_list_all_condi = []
    for ei, evnt in enumerate(event_type):
        for ci, condi in enumerate(condi_name): 
            print('  reading the evoked from disk')
            extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version +'_group_ave'
            avg_evo_fname = op.join(config_for_gogait.eeg_dir_GOODremove,
                                    config_for_gogait.base_fname_avg.format(**locals()))
           
            # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
            print("Output: ", avg_evo_fname)
            evoked_list = mne.read_evokeds(avg_evo_fname)         
            evoked_list_all_condi.append(evoked_list)
        
        
        # # Show condition names and baseline intervals
        # for e in evoked_list:
        #     print(f"Condition: {e.comment}, baseline: {e.baseline}")
            
        
        # evks = {'s2i10s8': evoked_list[0]}
        # evks['s2i10s8'].plot_psd(fmax = 60)
    
    #%%
    conds_all = []
    for ei, evnt in enumerate(event_type):
        for ci, condi in enumerate(condi_name): 
            conds = condi_name[ci] +'_' + event_type[ei] 
            conds_all.append(conds)
    
    conds_all = tuple(conds_all) 
            
            
    evks = dict(zip(conds_all, evoked_list_all_condi))
    
    
    
    #%%
    report = mne.Report()  
    plt.rcParams.update({'font.size': 20})
    fig1, ax1 = plt.subplots(figsize=(7,4))
    mne.viz.plot_compare_evokeds(dict(GOc_cue = evks['GOc_cue'], 
                                      GOu_cue = evks['GOu_cue'],
                                      NoGo_cue = evks['NoGo_cue']),
                                  picks="eeg", combine = 'gfp', 
                                  colors = ['g','b','r'], linestyles = ['solid','solid','solid'], 
                                  styles= {"GOc_cue": {"linewidth": 3},
                                           "GOu_cue": {"linewidth": 3}, 
                                           "NoGo_cue": {"linewidth": 3}},
                                  legend = False,
                                  truncate_yaxis=False, ylim=dict(eeg=[0, 6]),
                                  axes = ax1)
    # ax1.set_yticks([-10,-5, 0 ,5, 10], labels=[-10,-5, 0 ,5, 10])
    # ax1.set_xticks([-0.2, 0 ,0.2, 0.4, 0.6 ], labels=[-0.2, 0 ,0.2, 0.4, 0.6 ])
    ax1.set_xticks([])
    ax1.set_xlabel('')
    ax1.axvspan(t3min, t3max, facecolor='k', alpha = 0.1)
    ax1.axvspan(t1min, t1max, facecolor='k', alpha = 0.1)
    ax1.axvspan(t2min, t2max, facecolor='k', alpha = 0.1)
    # ax1.axvspan(t4min, t4max, facecolor='k', alpha = 0.1)
    plt.legend(['GOc', 'GOu', 'NoGo'], loc="upper right") # run, %matplotlib qt, to see the legend
    fig1.tight_layout()
    report.add_figure(fig1, title = 'GFP_at_cue', replace = True)
    # plt.savefig('GFP_at_cue.png', dpi=300)
    
    
    fig2, ax2 = plt.subplots(figsize=(7,4))
    mne.viz.plot_compare_evokeds(dict(GOc_target = evks['GOc_target'], 
                                      GOu_target = evks['GOu_target'],
                                      NoGo_target = evks['NoGo_target']),
                                  picks="eeg", combine = 'gfp', 
                                  colors = ['g','b','r'], linestyles = ['solid','solid','solid'],
                                  styles= {"GOc_target": {"linewidth": 3},
                                           "GOu_target": {"linewidth": 3}, 
                                           "NoGo_target": {"linewidth": 3}},
                                  legend = False,
                                  truncate_yaxis=False, ylim=dict(eeg=[0, 6]),
                                  axes = ax2)
    # ax2.set_yticks([-10,-5, 0 ,5, 10], labels=[-10,-5, 0 ,5, 10])
    # ax2.set_xticks([-0.2, 0 ,0.2, 0.4, 0.6 ], labels=[-0.2, 0 ,0.2, 0.4, 0.6 ])
    ax2.set_xticks([])
    ax2.set_xlabel('')
    ax2.axvspan(t1min, t1max, facecolor='k', alpha = 0.1)
    ax2.axvspan(t2min, t2max, facecolor='k', alpha = 0.1)
    ax2.axvspan(t3min, t3max, facecolor='k', alpha = 0.1)
    # ax2.axvspan(t4min, t4max, facecolor='k', alpha = 0.1)
    fig2.tight_layout()
    report.add_figure(fig2, title = 'GFP_at_target', replace = True)
    
    plt.close('all')    
       
    # finally saving the report after the for loop ends.     
    print('Saving the reports to disk') 
    report.title = 'Group GFP plots at cue and target across subs (n = ' + str(n_subs) +') ,' + version
    extension = 'GFP_plots'
    report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
    report.save(report_fname+'_cueTarget_' +version+ '_ppt.html', overwrite=True)
        
    
    
    # plt.savefig('GFP_at_target.png', dpi=300)
    
    # fig3, ax3 = plt.subplots(figsize=(10,6))
    # mne.viz.plot_compare_evokeds(dict(GOc_cue = evks['GOc_cue'], 
    #                                   GOc_target = evks['GOc_target']),   
    #                               picks="eeg", combine = 'gfp', 
    #                               colors = ['g','g'], linestyles = ['dotted', 'solid'],
    #                               styles= {"GOc_cue": {"linewidth": 3}, "GOc_target": {"linewidth": 3}},
    #                               legend = 'upper left',
    #                               truncate_yaxis=False, ylim=dict(eeg=[0, 6]),
    #                               axes = ax3)
    # ax3.axvspan(t1min, t1max, facecolor='k', alpha = 0.1)
    # ax3.axvspan(t2min, t2max, facecolor='k', alpha = 0.1)
    # ax3.axvspan(t3min, t3max, facecolor='k', alpha = 0.1)
    # ax3.axvspan(t4min, t4max, facecolor='k', alpha = 0.1)
    # fig3.tight_layout()
    
    # fig4, ax4 = plt.subplots(figsize=(10,6))
    # mne.viz.plot_compare_evokeds(dict(GOu_cue = evks['GOu_cue'], 
    #                                   GOu_target = evks['GOu_target']),   
    #                               picks="eeg", combine = 'gfp', 
    #                               colors = ['b','b'], linestyles = ['dotted', 'solid'],
    #                               styles= {"GOu_cue": {"linewidth": 3}, "GOu_target": {"linewidth": 3}},
    #                               legend = 'upper left',
    #                               truncate_yaxis=False, ylim=dict(eeg=[0, 6]),
    #                               axes = ax4)
    # ax4.axvspan(t1min, t1max, facecolor='k', alpha = 0.1)
    # ax4.axvspan(t2min, t2max, facecolor='k', alpha = 0.1)
    # ax4.axvspan(t3min, t3max, facecolor='k', alpha = 0.1)
    # ax4.axvspan(t4min, t4max, facecolor='k', alpha = 0.1)
    # fig4.tight_layout()
    
    
    # fig5, ax5 = plt.subplots(figsize=(10,6))
    # mne.viz.plot_compare_evokeds(dict(NoGo_cue = evks['NoGo_cue'], 
    #                                   NoGo_target = evks['NoGo_target']),   
    #                               picks="eeg", combine = 'gfp', 
    #                               colors = ['r','r'], linestyles = ['dotted', 'solid'],
    #                               styles= {"NoGo_cue": {"linewidth": 3}, "NoGo_target": {"linewidth": 3}},
    #                               legend = 'upper left',
    #                               truncate_yaxis=False, ylim=dict(eeg=[0, 6]),
    #                               axes = ax5)
    # ax5.axvspan(t1min, t1max, facecolor='k', alpha = 0.1)
    # ax5.axvspan(t2min, t2max, facecolor='k', alpha = 0.1)
    # ax5.axvspan(t3min, t3max, facecolor='k', alpha = 0.1)
    # ax5.axvspan(t4min, t4max, facecolor='k', alpha = 0.1)
    # fig5.tight_layout()
    
    # plt.rcParams.update({'font.size': 20})  
    
    # report.add_figure(fig1, title = 'Cue: gfp', replace = True)
    # report.add_figure(fig2, title = 'Target: gfp', replace = True)
    # report.add_figure(fig3, title = 'GOc: gfp', replace = True)
    # report.add_figure(fig4, title = 'GOu: gfp', replace = True)
    # report.add_figure(fig5, title = 'NoGo: gfp', replace = True)
    
    # #%% picking certain electrodes 
    # fig6, ax6 = plt.subplots(figsize=(10,6))
    # mne.viz.plot_compare_evokeds(dict(GOc_cue = evks['GOc_cue'], 
    #                                   GOu_cue = evks['GOu_cue'],
    #                                   NoGo_cue = evks['NoGo_cue']),
    #                               picks=['FCz'], combine = 'mean', 
    #                               colors = ['g','b','r'], linestyles = ['dotted','dotted','dotted'],
    #                               styles= {"GOc_cue": {"linewidth": 3},
    #                                        "GOu_cue": {"linewidth": 3}, 
    #                                        "NoGo_cue": {"linewidth": 3}},
    #                               truncate_yaxis=False, ylim=dict(eeg=[-4, 5]), 
    #                               legend = 'upper left',
    #                               show_sensors = False,
    #                               axes = ax6)
    # ax6.axvspan(t1min, t1max, facecolor='k', alpha = 0.1)
    # ax6.axvspan(t2min, t2max, facecolor='k', alpha = 0.1)
    # ax6.axvspan(t3min, t3max, facecolor='k', alpha = 0.1)
    # ax6.axvspan(t4min, t4max, facecolor='k', alpha = 0.1)
    # fig6.tight_layout()
    
    # fig7, ax7 = plt.subplots(figsize=(10,6))
    # mne.viz.plot_compare_evokeds(dict(GOc_target = evks['GOc_target'], 
    #                                   GOu_target = evks['GOu_target'],
    #                                   NoGo_target = evks['NoGo_target']),
    #                                  picks=['FCz'], combine = 'mean', 
    #                                  styles= {"GOc_target": {"linewidth": 3},
    #                                           "GOu_target": {"linewidth": 3}, 
    #                                           "NoGo_target": {"linewidth": 3}},
    #                                  colors = ['g','b','r'], linestyles = ['solid','solid','solid'],
    #                                  truncate_yaxis=False, ylim=dict(eeg=[-4, 5]), 
    #                                  legend = 'upper left',
    #                                  show_sensors = False,
    #                                  axes = ax7)
    # ax7.axvspan(t1min, t1max, facecolor='k', alpha = 0.1)
    # ax7.axvspan(t2min, t2max, facecolor='k', alpha = 0.1)
    # ax7.axvspan(t3min, t3max, facecolor='k', alpha = 0.1)
    # ax7.axvspan(t4min, t4max, facecolor='k', alpha = 0.1)
    # fig7.tight_layout()
    
    # fig8, ax8 = plt.subplots(figsize=(10,6))
    # mne.viz.plot_compare_evokeds(dict(GOc_cue = evks['GOc_cue'], 
    #                                   GOc_target = evks['GOc_target']),   
    #                              picks=['FCz'], combine = 'mean', 
    #                              colors = ['g','g'], linestyles = ['dotted', 'solid'],
    #                              styles= {"GOc_cue": {"linewidth": 3}, "GOc_target": {"linewidth": 3}},
    #                              truncate_yaxis=False, ylim=dict(eeg=[-4, 5]), 
    #                              legend = 'upper left',
    #                              show_sensors = False,
    #                              axes = ax8)
    # ax8.axvspan(t1min, t1max, facecolor='k', alpha = 0.1)
    # ax8.axvspan(t2min, t2max, facecolor='k', alpha = 0.1)
    # ax8.axvspan(t3min, t3max, facecolor='k', alpha = 0.1)
    # ax8.axvspan(t4min, t4max, facecolor='k', alpha = 0.1)
    # fig8.tight_layout()
    
    
    # fig9, ax9 = plt.subplots(figsize=(10,6))
    # mne.viz.plot_compare_evokeds(dict(GOu_cue = evks['GOu_cue'], 
    #                                   GOu_target = evks['GOu_target']),   
    #                              picks=['FCz'], combine = 'mean', 
    #                              colors = ['b','b'], linestyles = ['dotted', 'solid'],
    #                              styles= {"GOu_cue": {"linewidth": 3}, "GOu_target": {"linewidth": 3}},
    #                              truncate_yaxis=False, ylim=dict(eeg=[-4, 5]), 
    #                              legend = 'upper left',
    #                              show_sensors = False,
    #                              axes = ax9)
    # ax9.axvspan(t1min, t1max, facecolor='k', alpha = 0.1)
    # ax9.axvspan(t2min, t2max, facecolor='k', alpha = 0.1)
    # ax9.axvspan(t3min, t3max, facecolor='k', alpha = 0.1)
    # ax9.axvspan(t4min, t4max, facecolor='k', alpha = 0.1)
    # fig9.tight_layout()
    
    # fig10, ax10 = plt.subplots(figsize=(10,6))
    # mne.viz.plot_compare_evokeds(dict(NoGo_cue = evks['NoGo_cue'], 
    #                                   NoGo_target = evks['NoGo_target']),   
    #                               picks=['FCz'], combine = 'mean', 
    #                               colors = ['r','r'], linestyles = ['dotted', 'solid'],
    #                               styles= {"NoGo_cue": {"linewidth": 3}, "NoGo_target": {"linewidth": 3}},
    #                               truncate_yaxis=False, ylim=dict(eeg=[-4, 5]), 
    #                               legend = 'upper left',
    #                               show_sensors = False,
    #                               axes = ax10)
    # ax10.axvspan(t1min, t1max, facecolor='k', alpha = 0.1)
    # ax10.axvspan(t2min, t2max, facecolor='k', alpha = 0.1)
    # ax10.axvspan(t3min, t3max, facecolor='k', alpha = 0.1)
    # ax10.axvspan(t4min, t4max, facecolor='k', alpha = 0.1)
    # fig10.tight_layout()
    
    # plt.rcParams.update({'font.size': 20})
    
    
    # report.add_figure(fig6, title = 'Cue: mean', replace = True)
    # report.add_figure(fig7, title = 'Target: mean', replace = True)
    # report.add_figure(fig8, title = 'GOc: mean', replace = True)
    # report.add_figure(fig9, title = 'GOu: mean', replace = True)
    # report.add_figure(fig10, title = 'NoGo: mean', replace = True)
    
    
    # plt.close('all')    
       
    # # finally saving the report after the for loop ends.     
    # print('Saving the reports to disk')   
    # extension = 'compare_evoked_analysis'
    # report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
    # report.save(report_fname+'_cueTarget_' +version+ '.html', overwrite=True)
        