#Builtins
import logging
#Externals
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy import signal
#Customs
from modules import mth #DEVNOTE: Just add modules dir to virtual env pathlist to prevent import awkwardness

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

def linear_combine(arrs):
    '''
    Basic linear combination of image arrays

    Args:
        arrs  -- (i,y,x) ndarray or list of (y,x) ndarray : input image array

    Returns:
        carr  -- (y,x) ndarray : combined image array
    '''
    if type(arrs) is list: arrs = np.array(arrs)
    return np.average(arrs,axis=0)


def FCM(iarr, c, m, eps=.01, ml=100, verbose=True):
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

    Kwargs:
        eps  -- float : convergence threshold
        ml   -- int : max loops for algorithm before forced termination
        verbose -- boolean : controls text output

    Returns
        marr -- (y,x,c) ndarray : Image membership data
    '''

    #Initialize random membership array
    marr = np.random.rand(iarr.shape[0], iarr.shape[1], c)
    marr = np.einsum('yxc,yx->yxc', marr, np.sum(marr, axis=-1)**-1) #Norm probablities

    #Iterate
    i = 0
    converged = False
    if verbose: print("Iterating FCM...")
    while i<ml and not converged:
        i+=1
        if i%10 == 0 and verbose: print("%i iterations..."%i)

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
            if verbose: print("Reached convergence threshold, exiting iteration")
            converged = True

        marr = np.array(nmarr)

    return marr

def PFCM(iarr, c, m, gamma, imarr = None, eps=.01, ml=100, verbose=True, nbh = 1):
    '''
    Implements Fuzzy C-Means with an additional Penalty term which incorporates a spatial dependence
    This method originates from Yang and Huang 2007 [IMAGE SEGMENTATION BY FUZZY C-MEANS CLUSTERING ALGORITHM WITH A NOVEL PENALTY TERM]
    This function supercedes FCM() as setting nbh=0 is equivalent

    Args:
        iarr   -- (y,x) ndarray : Input image data
        c      -- int : number of clusters for segmentation
        m      -- int : fuzzification factor, 1 <= m < inf
        gamma  -- float : penalty weight factor

    Kwargs:
        eps      -- float : convergence threshold
        ml       -- int : max loops for algorithm before forced termination
        verbose  -- boolean : controls text output
        nbh      -- int > 0 : reach of pixel neighbourhood (1->8 pixels, 2->24 pixels...)
        imarr    -- (y,x,c) ndarray : If not None, use this array as initial fuzzy coefficients instead of a random array

    Returns
        marr -- (y,x,c) ndarray : Image membership data
    '''

    #Initialize membership array
    if imarr is not None:
        marr = imarr
    else:
        marr = np.random.rand(iarr.shape[0], iarr.shape[1], c)
        marr = np.einsum('yxc,yx->yxc', marr, np.sum(marr, axis=-1)**-1) #Norm probablities

    #Iterate
    i = 0
    converged = False
    if verbose: print("Iterating PFCM...")
    while i<ml and not converged:

        nmarr = np.array(marr)

        centroids = mth.calculate_centroids(iarr, marr, m)
        for j in range(c):
            #See eqn 13 Yang, Huang 2007
            p = 1/(m-1)
            nm = mth.neighbour_matrices(marr[...,j], nbh=nbh)
            npenalty = np.nansum((1-nm)**m, axis=0)
            num  = (iarr - centroids[j])**2 + gamma * npenalty
            dens = np.array([(iarr - centroids[rho])**2 + gamma * np.nansum((1-mth.neighbour_matrices(marr[...,rho], nbh=nbh))**m, axis=0) for rho in range(c)])
            fr   = np.sum((num/dens)**p, axis=0)
            nmarr[...,j] = fr**-1

        #Check convergence
        if np.average(np.abs(marr-nmarr)) < eps:
            if verbose: print("Reached convergence threshold, exiting iteration")
            converged = True
        marr = np.array(nmarr)

        i+=1
        if i%10 == 0 and verbose: print("%i iterations..."%i)
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

def sort_clusters(iarr, carr):
    '''
    Sorts clusters from lowest to highest average cell value, renumbering the clusters accordingly
    Useful for heatmap visualizations

    Args:
        iarr -- (y,x) ndarray : Image array values
        carr -- (y,x) ndarray : Image cluster values

    Returns:
        ocarr -- (y,x) ndarray : Renumbered cluster values
    '''

    c_IDs = np.sort(np.unique(carr))
    c_avgs = [np.average(iarr[carr==i]) for i in c_IDs]
    sorted_c_IDs = [y for x,y in sorted(zip(c_avgs,c_IDs), key=lambda p: p[0])]
    carr = carr + 1e7 #Temporarily renumber all clusters to unique intermediate IDs
    for i,j in zip(c_IDs, sorted_c_IDs):
        carr[carr==i+1e7] = j
    return carr














#
