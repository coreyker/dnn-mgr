import os, sys, glob, tables
import numpy as np
import theano
import cPickle
from pylearn2.utils import serial
from pylearn2.datasets.transformer_dataset import TransformerDataset
import GTZAN_dataset
import pdb

def aggregate_bof(model, dataset, save_file=None):
    """
    Create aggregate BOF features from output of MLP
    """
    if save_file is not None:
        if os.path.exists( save_file ):
            print save_file + ' already exists, aborting...'
            return ([],[])

    n_classes         = dataset.raw.y.shape[1]
    n_frames_per_file = dataset.raw.n_frames_per_file
    confusion         = np.zeros((n_classes, n_classes))
    n_examples        = len(dataset.raw.support)
    
    win_size          = n_frames_per_file // (30 // 5)
    hop_size          = win_size//2

    X     = model.get_input_space().make_theano_batch()
    Y     = model.fprop( X, return_all=True )
    fprop = theano.function([X],Y)

    batch_size = n_frames_per_file
    data_specs = dataset.raw.get_data_specs()
    iterator   = dataset.iterator(mode='sequential', 
        batch_size=batch_size, 
        data_specs=data_specs
        )

    i=0
    dataX = []
    dataY = []
    
    for el in iterator:

        # display progress indicator
        sys.stdout.write('Creating BOF: %2.0f%%\r' % (100*i/float(n_examples)))
        sys.stdout.flush()
    
        fft_data = el[0]
        feats    = fprop(fft_data)[-2] # the layer before softmax output layer

        labels = np.argmax(el[1], axis=1)
        true_label = labels[0]
        for entry in labels:
             assert entry == true_label # check for indexing prob

        mean = []
        std  = []
        for n in xrange(0, batch_size, hop_size):
            chunk = feats[n : n+win_size, :]
            mean.append(np.mean(chunk, axis=0))
            std.append(np.std(chunk, axis=0))                

        mean  = np.vstack(mean)
        std   = np.vstack(std)
        
        dataX.append( np.hstack((mean, std)) )
        dataY.append( np.ones( mean.shape[0] ) * true_label )
        
        i += batch_size

    print ''

    if save_file is not None:
        if not os.path.exists( save_file ):
            with open(save_file, 'w') as f:
                cPickle.dump((dataX, dataY), f, protocol=2 )
        else:
            print save_file + ' already exists, not saving'

    return (dataX, dataY)

if __name__ == "__main__":
        
    fold_file  = 'GTZAN_1024-fold-4_of_4.pkl'
    model_file = './saved-rlu-505050/mlp_rlu-fold-4_of_4.pkl' #'mlp_rlu_fold3_best.pkl'    

    # get model
    model = serial.load(model_file) 

    # get stanardized dictionary
    with open(fold_file) as f:
        config = cPickle.load(f)

    which_set = ['train', 'test', 'valid']
    for partition in which_set:
        save_file  = model_file.split('.pkl')[0] + '-' + partition + '-BOF.pkl'

        dataset = TransformerDataset(
            raw = GTZAN_dataset.GTZAN_dataset(config, partition),
            transformer = GTZAN_dataset.GTZAN_standardizer(config)
            ) 

        dataX, dataY = aggregate_bof(model, dataset, save_file)

