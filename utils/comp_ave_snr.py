import os
import numpy as np
import scikits.audiolab as audiolab

# calc ave snr
file_list = '/home/cmke/Datasets/_tzanetakis_genre/test_filtered.txt'

with open(file_list) as f:
	files = [l.strip() for l in f.readlines()]

base = '/home/cmke/Datasets/_tzanetakis_genre'

dir_list = ['/home/cmke/Datasets/_tzanetakis_F_500_RSD_allies', 
'/home/cmke/Datasets/_tzanetakis_F_500_RSD_random',
'/home/cmke/Datasets/_tzanetakis_F_500_RSD_jazz']

ign = 2048
snr = []
for d in dir_list:
	print d
	for i,f in enumerate(files):
		print 'iteration ', i
		x,_,_ = audiolab.wavread(os.path.join(base,f))
		xhat,_,_ = audiolab.wavread(os.path.join(d,f))
		
		L = min(len(x), len(xhat))

		snr.append(20*np.log10(np.linalg.norm(x[ign:L-ign-1])/np.linalg.norm(np.abs(x[ign:L-ign-1]-xhat[ign:L-ign-1])+1e-12)))

snr_ave = np.mean(snr)
print 'Ave snr: {}'.format(snr_ave)
