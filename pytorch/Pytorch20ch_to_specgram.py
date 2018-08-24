#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 22:31:39 2017

@author: ogawa
"""

import numpy as np
import matplotlib.mlab as mlab
import pandas as pd
import joblib



d = np.fromfile('no3.bin', dtype='<i2', count=-1)
N = 32
d = d.reshape((N, -1), order='F')

bat1 = pd.read_csv('./batxyz2.csv')

path = './data'


def make_specgram(current_loc):
    #    print('current_loc:', current_loc)

    #
    # bat1 xyz coordinates
    #
    current_time = current_loc * 2e-6  # 2 micro sec = 1 sample of 500kHz
    NNtime = (bat1.time - current_time).abs().idxmin()
    NNtime_pm1 = bat1.loc[NNtime - 1:NNtime + 1]  # NNtime-1 and NNtime

    #
    # Linear interpolation
    #
    bat1x = np.interp(current_time, NNtime_pm1.time, NNtime_pm1['bat(X)'])
    bat1y = np.interp(current_time, NNtime_pm1.time, NNtime_pm1['bat(Y)'])
    bat1z = np.interp(current_time, NNtime_pm1.time, NNtime_pm1['bat(Z)'])

    #
    # specgram
    #
    step = int((1 / 30.) / (1 / 500000.))  # 30fps 1frame vs 500kHz sampling
    step  # 1frame = 16666 sample points
    st = int(current_loc - step / 2)
    et = int(current_loc + step / 2)
    B_all = []
    for i in range(22):  # use ch.0 -- ch.21
        if i in [0, 10, 20, 21]:
            continue

        try:
            B, F, T = mlab.specgram(d[i, st:et],
                                    NFFT=128,
                                    Fs=500000,  # 500kHz
                                    window=mlab.window_hanning,
                                    noverlap=126
                                    )

            # get B[2:34, :] --> [32, 8270]
            B = B[2:34, :]

            B_all.append(B)
        except:
            pass
    B_all = np.dstack(B_all)  # 3D array
    B_all /= 40000  # ad-hoc normalizatoin
    print('current_loc:', current_loc, [B_all.max(), B_all.min()], [bat1x, bat1y, bat1z])

    np.save(path + "/trueXYZ_"+ '{:09d}'.format(current_loc),np.array([bat1x, bat1y, bat1z]))
    np.save(path + "/specgram_" + '{:09d}'.format(current_loc),B_all)




results = joblib.Parallel(n_jobs=-1)(
    [joblib.delayed(make_specgram)(current_loc) for current_loc in np.arange(1606200, 46546200, 1500)])
