import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy import signal
from modules import mth

#------
# This module contains functions pertaining to Image Processing
# i.e. functions which operate on and typically return 2D gridded data
#------


def FWT(iarr, d = 3, wl='db1', FAR = 1):
    '''
    Performs a variation on a fuzzy wavelet transform process on the input 2D data

    Args:
        iarr -- (y,x) ndarray : Input image data
    Kwargs:
        d   -- int : Number of DWT decompositions applied in denoising
        wl  -- str : Wavelet for DWT basis
        FAR -- int : Optional local averaging pixel span, 1<=FAR<inf

    Returns:
        row_im, col_im -- (y,x) ndarrays : Smoothed row/column DWT outputs
    '''

    #Basic input check
    if type(iarr) is not np.ndarray:
        print("\nError : Input object is not a numpy array\n")
        raise Exception
    elif len(iarr.shape) != 2:
        print('\nError : Input array object does not have dimensionality 2')
        print('Dimensionality: %i\n'%len(iarr.shape))
        raise Exception

    #Denoise image -- Apply 1D row-column DWT, then iDWT setting coefficients of all but last decomposition level to 0
    row_cf = pywt.wavedec(iarr, wl, level=d, axis = 0)
    col_cf = pywt.wavedec(iarr, wl, level=d, axis = 1)

    for i,cf in enumerate(row_cf):
        if i > 1: cf.fill(0)
    for i,cf in enumerate(col_cf):
        if i > 1: cf.fill(0)

    row_im = pywt.waverec(row_cf, wl, axis = 0)
    col_im = pywt.waverec(col_cf, wl, axis = 1)

    #Perform a local averaging on the resulting images, then combine linearly
    avg = np.ones((FAR, FAR))
    row_im = signal.convolve2d(row_im, avg)
    col_im = signal.convolve2d(col_im, avg)

    return row_im, col_im

def FCM(iarr, c, m, eps=.01, ml=100):
    '''
    Implements standard Fuzzy C-Means algorithm on an input array, returning probabalistic segment memberships
    Note the limitations of naive FCM:
        Does not account for spatial correlation
            -thus is sensitive to noise/speckling
        Requires input number of clusters
        Iterates from random initial condition

    Args:
        iarr -- (y,x) ndarray : Input image data
        c    -- int : number of clusters for segmentation
        m    -- int : fuzzification factor, 1 <= m < inf
        eps  -- float : convergence threshold
        ml   -- int : max loops for algorithm before forced termination

    Returns
        marr -- (y,x,c) ndarray : Image membership data
    '''

    #Basic input check
    if type(iarr) is not np.ndarray:
        print("\nError : Input object is not a numpy array\n")
        raise Exception
    elif len(iarr.shape) != 2:
        print('\nError : Input array object does not have dimensionality 2')
        print('Dimensionality: %i\n'%len(iarr.shape))
        raise Exception

    #Initialize random membership array
    marr = np.random.rand(iarr.shape[0], iarr.shape[1], c)
    marr = np.einsum('yxc,yx->yxc', marr, np.sum(marr, axis=-1)) #Norm probablities

    #Iterate
    i = 0
    converged = False
    print("Iterating FCM...")
    while i<ml and not converged:
        i+=1
        if i%10 == 0: print("%i iterations..."%i)

        #New membership array object
        nmarr = np.array(marr)

        centroids = mth.calculate_centroids(iarr, marr, m)
        for j in range(c):
            p = 2/(m-1)
            num  = np.abs(iarr - centroids[j])
            dens = np.abs([iarr - centroids[rho] for rho in range(c)])
            fr   = np.sum((num/dens)**p, axis=0)
            nmarr[...,j] = fr**-1

        #Check convergence
        if np.average(np.abs(marr-nmarr)) < eps:
            print("Reached convergence threshold, exiting iteration")
            converged = True

        marr = np.array(nmarr)

    return marr

def naive_membership_assign(marr):
    '''
    Extremely basic membership assignment for probabalistic clustering
    Simply assigns to segment with highest membership fraction

    Args:
        marr -- (y,x,c) ndarray : Membership array

    Returns:
        sarr -- (y,x) ndarray : Segmented array
    '''

    return np.argmax(marr, axis=-1)


















#
