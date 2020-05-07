from modules import io
from modules import ip
import matplotlib.pyplot as plt



tfile = "data/NSDIC_0051/nt_20200502_f18_nrt_s.bin"
header, arr = io.read_NSIDC0051_file(tfile)

rw,cl = ip.FWT(arr)                       #Smooth noise with wavelet transform
marr = ip.FCM(rw, 7, 5, eps=.0001)    #Standard Fuzzy C-Means probabalistic membership assignment
sarr = ip.naive_membership_assign(marr)     #Naive deterministic membership assignment

plt.imshow(sarr, cmap="hsv")
plt.savefig('sarr1')
plt.imshow(-1*sarr, cmap="Blues")
plt.savefig('sarr2')











#
