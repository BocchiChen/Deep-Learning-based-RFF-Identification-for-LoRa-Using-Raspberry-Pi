# -*- coding: utf-8 -*-

import numpy as np
import builtins
import matplotlib as mb
from rtlsdr import RtlSdr
from pylab import *
from rtlsdr import *
from scipy import signal
import math
import cmath
import time
import matplotlib.pyplot as plt
import scipy.signal


def receivePackets(sampling_times, sampling_rate, sampling_length, sdr_gain, carrier_rate):

    sdr = RtlSdr()
    # Configure sdr device
    sdr.sample_rate = sampling_rate * 1e6
    sdr.center_freq = carrier_rate * 1e6 # MHz
    #sdr.freq_correction = 20
    sdr.set_agc_mode(False)
    sdr.gain = sdr_gain
    # Set LoRa configuration parameters
    fc = carrier_rate * 1e6
    bandwidth = 125e3
    SF = 7
    fs = sampling_rate * 1e6
    T = 2**SF/bandwidth
    Ts = 1/fs
    L = int(2**SF/(bandwidth*Ts))
    symbols = 8
    sample_length = sampling_length
    asize = 0
    cfo = []
    Sxc = []

    # ----------------------------------------------------------------------------------------------------------------------------
    IQlist = []
    for times in range(0, sampling_times):
        samples = sdr.read_samples(sample_length)
        IQlist.append(samples)
        time.sleep(0.13)
    sdr.close()

    IQmat = np.array(IQlist)
    IQvalue = abs(IQmat)

    lora_array = np.zeros((times*5, sample_length), dtype=complex)
    row = 0
    column = 0

    for x in range(0, times):
        for y in range(0, sample_length-1):
            if IQvalue[x][y] > 1:
                lora_array[row][column] = IQmat[x][y]
                column += 1
                if lora_array[row][0] != 0 and lora_array[row][symbols*L] != 0:
                    row += 1
                    column = 0
        column = 0
    #lora.L = 1024 = 1.024ms
    #IQvalue[x][y+1] < 0.2
    # ----------------------------------------------------------------------------------------------------------------------------
    for times in range(0, row):

        lora_samples = lora_array[times][0:].flatten()

        # Method to calculate P(d)
        n_set = np.arange(0, 10)
        P1 = np.zeros(len(n_set), dtype=complex)
        for i, n in enumerate(n_set):
            P1[i] = builtins.sum(lora_samples[n+k].conj()
                                 * lora_samples[n+k+L] for k in range(L))

        # Method to calculate R(d)
        R1 = np.zeros(len(n_set), dtype=complex)
        for i, n in enumerate(n_set):
            R1[i] = builtins.sum(lora_samples[n+k+L].conj()
                                 * lora_samples[n+k+L] for k in range(L))

        M = abs(P1)/R1

        # Find the preambles with M[n]
        t = 0
        coarse_preamble = None
        while t < len(n_set):
            if M[t] > 0.93:  # threshold
                coarse_preamble = lora_samples[t:t+9*L]
                break
            t += 1

        if coarse_preamble is None:
            continue

        # Fine synchronization algorithm
        f_ideal = np.zeros(L)
        for n in range(L):
            f_ideal[n] = - bandwidth/2 + bandwidth * n/(T * fs)

        f_r_phase = np.unwrap(np.angle(coarse_preamble))
        f_r = (np.diff(f_r_phase)/(2.0*np.pi) * fs)

        cross_cor = np.zeros(L)
        for i in range(0, 20):
            cross_cor[i] = builtins.sum(f_ideal[n]*f_r[n+i] for n in range(L))

        ind = np.argmax(cross_cor)

        if ind > 10:
            continue

        preamble_ts = coarse_preamble[ind:ind+8*L]

        # ----------------------------------------------------------------------------------------------------------------------------
        # CFO Estimation Algorithm
        # Coarse CFO Estimation
        
        # Calculate the instantaneous frequency
        f_p_phase = np.unwrap(np.angle(preamble_ts))
        f_p = (np.diff(f_p_phase)/(2*np.pi) * fs)

        Delta_f_coarse = (1/L) * builtins.sum(f_p[n] for n in range(L))

        preamble_ts_cfo = np.zeros(8*L, dtype=complex)
        for i in range(len(preamble_ts)):
            preamble_ts_cfo[i] = preamble_ts[i] * cmath.exp(-2.0*np.pi*Delta_f_coarse*1/1e6*i*1j)

        # Fine CFO Estimation
        Delta_f_fine = - (1e6)/(2.0*np.pi*L) * np.angle(builtins.sum(preamble_ts_cfo[n]*preamble_ts_cfo[n+L].conj() for n in range(L)))
        for i in range(len(preamble_ts_cfo)):
            preamble_ts_cfo[i] = preamble_ts_cfo[i] * cmath.exp(-2.0*np.pi*Delta_f_fine*1/1e6*i*1j)
        
        # Save CFOs
        cfo.append(Delta_f_coarse+Delta_f_fine)

        if asize == 1100:  # 0-999,1000-1099
            break
        asize += 1

        # Represent I/Q samples as spectrograms
        f, t, Sxx = signal.spectrogram(preamble_ts_cfo, fs=fs, window='boxcar', nperseg=256, noverlap=128, nfft=256, return_onesided=False,mode='magnitude')
        plt.pcolormesh(t, scipy.fft.fftshift(f), scipy.fft.fftshift(Sxx, axes=0), shading='gouraud')
        plt.axis('off')
        plt.savefig('C:/Users/13620/Desktop/spec'+str(asize)+'.png')
        Sxc.append(np.array(Sxx))       
         
        # Auto save
        # 1100 preambles
    Sxca = np.array(Sxc)
    np.save(r'./npydata/test/test1.npy', Sxca)
