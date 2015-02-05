import os
import numpy as np

freqs = np.hstack(([20], np.arange(1000,12000,1000)))
for f in freqs:
	print 'On cutoff: {}'.format(f)
	os.system('''python utils/filtered_classify.py --dnn_model saved_models/dnn/S_500_RSD.pkl --aux_model saved_models/rf/S_500_RSD_AF_LAll.pkl --data_dir /home/cmke/Datasets/_tzanetakis_S_500_RSD_random/ --test_list gtzan/test_stratified.txt --filter_cutoff {hz:d} --dnn_save_file /home/cmke/Datasets/_tzanetakis_S_500_RSD_random/__dnn__/S_500_RSD-{hz:d}.txt --aux_save_file /home/cmke/Datasets/_tzanetakis_S_500_RSD_random/__rf__/S_500_RSD_AF_LAll-{hz:d}.txt'''.format(hz=f))



