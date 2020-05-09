from modules import io
from modules import ip
import matplotlib.pyplot as plt

tfile = "data/NSDIC_0051/nt_20200502_f18_nrt_s.bin"
header, arr = io.read_NSIDC0051_file(tfile)

rw,cl = ip.FWT(arr)                       #Smooth noise with wavelet transform
marr = ip.PFCM(rw, 7, 4, 50, eps=.0001, ml=50)    #Penalized Fuzzy C-Means probabalistic membership assignment
#marr = ip.FCM(rw, 7, 4, eps=.0001)    #Standard Fuzzy C-Means probabalistic membership assignment
sarr = ip.naive_membership_assign(marr)     #Naive deterministic membership assignment

plt.imshow(sarr, cmap="hsv")
plt.savefig('sarr3')
plt.imshow(-1*sarr, cmap="Blues")
plt.savefig('sarr4')











#
