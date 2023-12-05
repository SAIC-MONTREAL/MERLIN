'''
Process reformatted data from evaluate.py. Save data as csv and plot data against timeseries.
'''

import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import argparse
import os, glob
import pandas as pd
import numpy as np

def plot_data(all_data, dir):
    '''
    Plot data from each pkl file separately.
    '''
    for df in all_data:
        plt.figure()
        idx = 1
        for col in df.columns:
            fig = plt.subplot(3,4,idx)
            x = df.index
            y = df[col]
            plt.plot(x,y,label=df.name)
            plt.title(col, fontsize=8)
            plt.tick_params(axis='both', which='major', labelsize=6)
            plt.tick_params(axis='both', which='minor', labelsize=4)
            fig.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            fig.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            idx += 1
            # Some bandaid solution to avoid trying to plot load of all individual cell
            if idx > 11:
                break
        
        plt.subplots_adjust(hspace=0.5, wspace=0.4)
        plt.savefig(os.path.join(dir, df.name+'.eps'), format='eps')
    
    plot_all_data_overlay(all_data, dir)

def plot_all_data_overlay(all_data, dir):
    '''
    Overlay plot of data from all pkl files.
    '''
    plt.figure()    
    idx = 1
    for col in all_data[0].columns:
        fig = plt.subplot(3,4,idx)
        for df in all_data:
            x = df.index
            y = df[col]
            plt.plot(x,y,label=df.name)
            plt.tick_params(axis='both', which='major', labelsize=6)
            plt.tick_params(axis='both', which='minor', labelsize=4)
            fig.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            fig.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.title(col, fontsize=8)
        idx += 1
        # Some bandaid solution to avoid trying to plot load of all individual cell
        if idx > 11:
            break
        
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.savefig(os.path.join(dir, 'overlay.eps'), format='eps')