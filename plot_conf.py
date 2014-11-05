import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

import sys
import numpy as np
import theano
from pylearn2.utils import serial
from pylearn2.datasets.transformer_dataset import TransformerDataset
import cPickle
import GTZAN_dataset

from test_mlp_script import frame_misclass_error, file_misclass_error

def plot_conf_mat(conf, ax_labels):
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	ax.set_aspect(1)
	ax.imshow(np.array(conf), cmap=plt.cm.gray_r, interpolation='nearest')

	width,height = conf.shape
	for x in xrange(width):
	    for y in xrange(height):
	    	if conf[x][y]<50:
	    		color='k'
	    	else:
	    		color='w'
	        ax.annotate('%2.1f'%conf[x][y], xy=(y, x), horizontalalignment='center', verticalalignment='center',color=color)

	ax.xaxis.tick_top()
	plt.xticks(range(width), ax_labels)
	plt.yticks(range(height), ax_labels)

	labels = ax.get_xticklabels() 
	for label in labels: 
	    label.set_rotation(30) 

	plt.show()


#model_files = ['./saved-rlu-505050/mlp_rlu_fold1_best.pkl', './saved-rlu-505050/mlp_rlu_fold2_best.pkl', './saved-rlu-505050/mlp_rlu_fold3_best.pkl', './saved-rlu-505050/mlp_rlu_fold4_best.pkl']
model_files = ['./saved/mlp_rlu-fold-1_of_4.pkl', './saved/mlp_rlu-fold-2_of_4.pkl', './saved/mlp_rlu-fold-3_of_4.pkl', './saved/mlp_rlu-fold-4_of_4.pkl']
fold_files  = ['GTZAN_1024-fold-1_of_4.pkl', 'GTZAN_1024-fold-2_of_4.pkl', 'GTZAN_1024-fold-3_of_4.pkl', 'GTZAN_1024-fold-4_of_4.pkl'] 

fold_err  = len(model_files) * [None]
fold_conf = len(model_files) * [None]

for i, (model_file, fold_file) in enumerate(zip(model_files, fold_files)):

    model = serial.load(model_file)  

    # get stanardized dictionary  
    which_set = 'test'
    with open(fold_file) as f:
        config = cPickle.load(f)
    
    dataset = TransformerDataset(
        raw = GTZAN_dataset.GTZAN_dataset(config, which_set),
        transformer = GTZAN_dataset.GTZAN_standardizer(config)
        )

    # test error
    #err, conf = frame_misclass_error(model, dataset)    
    fold_err[i], fold_conf[i] = file_misclass_error(model, dataset)

# plot confusions error
fold_conf = np.array(fold_conf, dtype=np.float32)

ave_err = np.mean(fold_err)
std_err = np.std(fold_err)

ave_conf = np.mean(fold_conf/25.*100, axis=0)
std_conf = np.std(fold_conf/25.*100, axis=0)

print ave_err
print std_err

ax_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
plot_conf_mat(ave_conf, ax_labels)

# run test_mlp_script.py GTZAN_1024-filtered-fold.pkl ./saved-rlu-505050/mlp_rlu_filtered_fold_best.pkl
# filt_err = err
# filt_conf = conf
# ave_filt_conf = filt_conf/np.sum(filt_conf, axis=1) * 100
# plot_conf_mat(ave_filt_conf, ax_labels)

#plt.savefig('confusion_matrix_deflated.pdf', format='pdf')
