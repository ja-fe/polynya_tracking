from modules import io
from modules import ip
import matplotlib.pyplot as plt
import gdal
import numpy as np
import osr

#-------------
# AVHRR Notes:
# AVHRR has both very high (1.1km at nadir) resolution and fast revisit times (avg 14 orbits per sattelite per day)
#-------------


#Load the sample AVHRR data, reprojecting to polar sterographic
dataset = gdal.Warp('AVHRRps.tif','data/AVHRR/HRPT/NSS.HRPT.NN.D20122.S2350.E0002.B7704545.GC', dstSRS='EPSG:3411')
print("AVHRR: Number of Bands -- ", dataset.RasterCount)      # how many bands, to help you loop
print("AVHRR: Raster size: (%i,%i)"%(dataset.RasterXSize, dataset.RasterYSize))      # how many columns

#BAND INFO
# Band 1  (Reds) Common uses: urban, vegetation, snow/ice, daytime cloud studies
# Band 2  (Near Infrared) Common uses: vegetation, land/water boundaries, snow/ice, flooding studies
# Band 3A (Mid Infrared) Common uses: Vegetation, snow/ice, dust monitoring
# Band 3B (Thermal1) Common uses: Surface temperature, wildfire detection, nighttime clouds, volcanic activity
# Band 4  (Thermal2) Common uses: Surface temperature, wildfire detection, nighttime clouds, volcanic activity
# Band 5  (Thermal3) Common uses: Sea surface temperature, water vapor path radiance

band = dataset.GetRasterBand(5)
arr = band.ReadAsArray()

#To save on computation power for testing, select subregion
#  Select Northern Water polynya area (NOW) manually for this file
#  @DEVNOTE: In future, parse georeferencing metadat to crop automatically
arr = arr[500:850,1100:1500]

rw,cl = ip.FWT(arr,d=2)                                  #Smooth noise with wavelet transform
marr = ip.PFCM(rw, 3, 6, 50, nbh=1, eps=.001, ml=200)    #Penalized Fuzzy C-Means probabalistic membership assignment
sarr = ip.naive_membership_assign(marr)                  #Naive deterministic membership assignment

plt.imshow(-1*sarr,cmap="Blues")
plt.savefig('analyses/AVHRR/HRPT/Band5Sample')




#
