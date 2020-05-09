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

def neighbour_matrices(iarr, nbh = 1):
    '''
    Returns an array of shifted matrices with the ijth element in each matrix being the value of one
        of the neighbours of the ijth element in the input
    Mask arrays are a more succint alternative but quickly become massive for large datasets
    np.nan objects are used to denote a lack of neighbour (e.g. no neighbour to the right for a right edge pixel)

    Args:
        iarr -- (y,x) ndarray : Input data array

    Kwargs:
        nbh -- int>0 : Span of neighbourhood

    Returns:
        matrices -- (n,y,x) ndarray : neighbour values array
    '''

    matrices = []
    for i in range(-nbh, nbh+1):
        for j in range(-nbh, nbh+1):
            if i == 0 and j ==0: continue
            nm = np.roll(iarr, i, axis = 0)
            nm = np.roll(nm  , j, axis = 1)

            #Edge effects
            if i>0: nm[:i]   = np.nan
            if i<0: nm[i:]   = np.nan
            if j>0: nm[:,:j] = np.nan
            if j<0: nm[:,j:] = np.nan

            matrices.append(nm)

    return np.array(matrices)














#
