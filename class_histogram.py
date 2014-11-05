import sys
import numpy as np
import theano
from pylearn2.utils import serial
from pylearn2.datasets.transformer_dataset import TransformerDataset
import cPickle
import GTZAN_dataset

import pdb

def class_histogram(model, dataset):
    """
    Function to compute the file-level classification error by classifying
    individual frames and then voting for the class with highest cumulative probability
    """
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop( X )
    fprop = theano.function([X],Y)

    n_classes  = dataset.raw.y.shape[1]
    confusion  = np.zeros((n_classes, n_classes))
    n_examples = len(dataset.raw.support)
    n_frames_per_file = dataset.raw.n_frames_per_file

    batch_size = n_frames_per_file
    data_specs = dataset.raw.get_data_specs()
    iterator = dataset.iterator(mode='sequential', 
        batch_size=batch_size, 
        data_specs=data_specs
        )

    i=0
    histogram = []
    for el in iterator:

        # display progress indicator
        sys.stdout.write('Classify progress: %2.0f%%\r' % (100*i/float(n_examples)))
        sys.stdout.flush()
    
        fft_data     = np.array(el[0], dtype=np.float32)
        frame_labels = np.argmax(fprop(fft_data), axis=1)
        hist         = np.bincount(frame_labels, minlength=n_classes)
        histogram.append(hist)

        i += batch_size

    return histogram

#if __name__ == '__main__':
    
#    _, fold_file, model_file = sys.argv
fold_file = 'GTZAN_1024-fold-1_of_4.pkl'
model_file = './saved-rlu-505050/mlp_rlu_fold1_best.pkl'

# get model
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

hist = class_histogram(model, dataset)
hist = np.vstack(hist)

test_files = np.array(config['test_files'])
test_labels = test_files//100

most_votes = np.argmax(hist,axis=0)
most_rep_files = test_files[most_votes]
most_rep_hist = hist[most_votes, :]

prediction = np.argmax(hist, axis=1)
top_pred = np.argsort(hist, axis=1)
top_pred = top_pred[:,-1::-1]

err_list = []

# for i, (l,p) in enumerate(zip(test_labels, prediction)):
#     if l != p:
#         err_list.append(i)

for i, (l,p) in enumerate(zip(test_labels, top_pred)):    
    if l not in p[:3]:
        err_list.append(i)


ax_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

n = err_list[0]
err_file = test_files[n]
err_hist = hist[n] 














