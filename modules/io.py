import numpy as np
from array import array
import matplotlib.pyplot as plt

#------
# This module handles the reading and writing of file structures
#------

def read_NSIDC0051_file(fpath):
    '''
    Reads in an individual National Snow & Ice Data Centre (NSIDC) Sea ice concentration (SIC) file
    Assumes the file follows NSIDC naming conventions
    Full dataset docs [https://nsidc.org/data/NSIDC-0051/versions/1]

    Args:
        fpath  -- str : File location

    Returns:
        header -- (x,)  np,ndarray : unprocessed binary file header
        darray -- (x,y) np.ndarray : Sea ice concentration raster data values
    '''

    #NSIDC files use a 300 byte header followed by one-byte raster cell values
    rawdata = np.fromfile(fpath, dtype=np.uint8)
    header = rawdata[:300]

    #Northern or southern hemisphere grid?
    if fpath[fpath.index('.')-1] == 'n':
        darray = rawdata[300:].reshape((448,304))
    elif fpath[fpath.index('.')-1] == 'n':
        darray = rawdata[300:].reshape((332,316))

    darray = darray.astype(np.float32)
    return header, darray


def process_NSIDC0051_header():
    '''
    TODO: parse file metadata into dataset class object
    '''
    pass
