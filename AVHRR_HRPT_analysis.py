#Builtins
import logging
import sys
#Externals
import matplotlib.pyplot as plt
import gdal
import numpy as np
import scipy.signal as sgnl
#Customs
from modules import io
from modules import ip

#-------------
# AVHRR Notes:
# AVHRR has both very high (1.1km at nadir) resolution and fast revisit times (avg 14 orbits per sattelite per day)
#-------------

#Begin log
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("AVHRR.log"),   #Log to file
        logging.StreamHandler()             #And to console
    ]
)
logging.info('\nBeginning analysis of AVHRR data')

#Load the sample AVHRR data, reprojecting to polar sterographic
dataset = gdal.Warp('AVHRRps.tif','data/AVHRR/HRPT/NSS.HRPT.NN.D20122.S2350.E0002.B7704545.GC', dstSRS='EPSG:3411')
logging.info("AVHRR: Number of Bands -- %i"%dataset.RasterCount)
logging.info("AVHRR: Raster size: (%i,%i)"%(dataset.RasterXSize, dataset.RasterYSize))

#BAND INFO
# Band 1  (Reds) Common uses: urban, vegetation, snow/ice, daytime cloud studies
# Band 2  (Near Infrared) Common uses: vegetation, land/water boundaries, snow/ice, flooding studies
# Band 3A (Mid Infrared) Common uses: Vegetation, snow/ice, dust monitoring
# Band 3B (Thermal1) Common uses: Surface temperature, wildfire detection, nighttime clouds, volcanic activity
# Band 4  (Thermal2) Common uses: Surface temperature, wildfire detection, nighttime clouds, volcanic activity
# Band 5  (Thermal3) Common uses: Sea surface temperature, water vapor path radiance

#I find thermal bands are ideal for single-band segmentation because polynyas are identifyable by the heat produced
#  through the latent heat of fusion/ sublimation heat separating them from their boundaries. Also thermal bands ignore
#  cloud cover much more effectively.
band = dataset.GetRasterBand(4)
arr = band.ReadAsArray()

#To save on computation power for testing, select subregion
#  Select Northern Water polynya area (NOW) manually for this file
#  @DEVNOTE: In future, parse georeferencing metadata to crop automatically
arr = arr[500:850,1100:1500]

denoised_arrs = ip.FWT(arr,d=3)                                  #Smooth noise with wavelet transform
rw,cl = [sgnl.medfilt(i,7) for i in denoised_arrs]

#Save an array/image file every iteration to study convergence of PFCM
marr = np.random.rand(rw.shape[0], rw.shape[1], 9) #Random initial membership array
for i in range(100):
    marr = ip.PFCM(rw, 9, 6, 50, imarr = marr, nbh=0, eps=.0001, ml=1)    #Penalized Fuzzy C-Means probabalistic membership assignment
    sarr = ip.naive_membership_assign(marr)                               #Naive deterministic membership assignment
    sarr = ip.sort_clusters(arr, sarr)                                    #Reorder cluster IDs based on average value of cluster
    np.save("analyses/AVHRR/HRPT/with_penalty/npfiles/s%iiters"%i, sarr)  #Save segmented array
    np.save("analyses/AVHRR/HRPT/with_penalty/npfiles/m%iiters"%i, marr)  #Save fuzzy membership array
    plt.imshow(-1*sarr,cmap="Blues")                                      #Save a snapshot of segmentation
    plt.savefig('analyses/AVHRR/HRPT/with_penalty/ims/%iiters'%i)

plt.imshow(-1*sarr,cmap="Blues")
plt.savefig('analyses/AVHRR/HRPT/full9')

logging.info('Completed')



#
