import numpy as np

#------
# This module contains helper math functions
#------

def calculate_centroids(iarr, marr, m):
    '''
    Calculates the cluster centroids associated with a fuzzy membership array

    Args:
        iarr -- (y,x) ndarray : image array
        marr -- (y,x,c) ndarray : cluster membership array
        m    -- int : fuzzification factor, 1 <= m < inf

    Returns:
        centroids -- (c) ndarray : cluster centroid values
    '''

    centroids = np.zeros(marr.shape[-1])
    for c in range(marr.shape[-1]):
        w = marr[...,c]**m
        cn = np.sum(w*iarr)/np.sum(w)
        centroids[c] = (cn)

    return centroids















#
