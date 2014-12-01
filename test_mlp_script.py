import sys, re, cPickle
import numpy as np
import theano

from pylearn2.utils import serial
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import pylearn2.config.yaml_parse as yaml_parse
import GTZAN_dataset

import pdb
def frame_misclass_error(model, dataset):
    """
    Function to compute the frame-level classification error by classifying
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
    for el in iterator:

        # display progress indicator
        sys.stdout.write('Classify progress: %2.0f%%\r' % (100*i/float(n_examples)))
        sys.stdout.flush()
    
        fft_data    = np.array(el[0], dtype=np.float32)
        vote_labels = np.argmax(fprop(fft_data), axis=1)
        true_labels = np.argmax(el[1], axis=1)

        for l,v in zip(true_labels, vote_labels):
            confusion[l, v] += 1

        i += batch_size

    total_error = 100*(1 - np.sum(np.diag(confusion)) / np.sum(confusion))
    print ''
    return total_error, confusion

def file_misclass_error(model, dataset):
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
    n_frames_per_file   = dataset.raw.n_frames_per_file
    n_frames_per_sample = dataset.raw.n_frames_per_sample

    batch_size = n_frames_per_file // n_frames_per_sample
    data_specs = dataset.raw.get_data_specs()
    #data_specs = (CompositeSpace((VectorSpace(dim=513*n_frames_per_sample), VectorSpace(dim=n_classes))), ("features", "targets")) 
    iterator   = dataset.iterator(mode='sequential', 
        batch_size=batch_size, 
        data_specs=data_specs
        )
    
    i=0
    for el in iterator:

        # display progress indicator
        sys.stdout.write('Classify progress: %2.0f%%\r' % (100*i/float(n_examples)))
        sys.stdout.flush()
    
        fft_data     = np.array(el[0], dtype=np.float32)
        frame_labels = np.argmax(fprop(fft_data), axis=1)
        hist         = np.bincount(frame_labels, minlength=n_classes)
        vote_label   = np.argmax(hist) # most used label

        labels = np.argmax(el[1], axis=1)
        true_label = labels[0]
        for entry in labels:
             assert entry == true_label # check for indexing prob

        confusion[true_label, vote_label] += 1

        i+=batch_size

    total_error = 100*(1 - np.sum(np.diag(confusion)) / np.sum(confusion))
    print ''
    return total_error, confusion

def file_misclass_error_topx(model, dataset, topx=3):
    """
    Function to compute the file-level classification error by classifying
    individual frames and then voting for the class with highest cumulative probability

    Check topx most probable results
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

    hits = 0
    n = 0
    i=0        
    for el in iterator:

        # display progress indicator
        sys.stdout.write('Classify progress: %2.0f%%\r' % (100*i/float(n_examples)))
        sys.stdout.flush()
    
        fft_data     = np.array(el[0], dtype=np.float32)
        frame_labels = np.argmax(fprop(fft_data), axis=1)
        hist         = np.bincount(frame_labels, minlength=n_classes)
        vote_label   = np.argsort(hist)[-1:-1-topx:-1] # most used label

        labels = np.argmax(el[1], axis=1)
        true_label = labels[0]
        for entry in labels:
             assert entry == true_label # check for indexing prob

        if true_label in vote_label:
            hits+=1

        n+=1
        i+=batch_size

    print ''
    return hits/float(n)*100


def pp_array(array): # pretty printing
    for row in array:
        print ['%04.1f' % el for el in row]


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description='''Script to test DNN. Measure framelevel accuracy. 
        Option to use a majority vote for over the frames in each test recording.
        ''')

    parser.add_argument('model_file', help='Path to trained model file')
    parser.add_argument('--testset', help='Optional. If not specified, the testset from the models yaml src will be used')
    parser.add_argument('--majority_vote', action='store_true', help='Measure framelevel accuracy with ')
    args = parser.parse_args()
    
    # get model
    model = serial.load(args.model_file)  

    if args.testset: # dataset config passed in from command line
        print 'Using dataset passed in from command line:'
        with open(args.testset) as f: cfg = cPickle.load(f)
        dataset = TransformerDataset(
            raw=GTZAN_dataset.GTZAN_dataset(which_set='test', config=cfg),
            transformer=GTZAN_dataset.GTZAN_standardizer(config=cfg)
            )

    else: # get dataset from model's yaml_src
        print "Using dataset from model's yaml src:"
        p = re.compile(r"which_set.*'(train)'")
        dataset_yaml = p.sub("which_set: 'test'", model.dataset_yaml_src)
        dataset = yaml_parse.load(dataset_yaml)

    # measure test error
    if args.majority_vote:
        print 'Using majority vote'
        err, conf = file_misclass_error(model, dataset)
    else:
        print 'Not using majority vote'
        err, conf = frame_misclass_error(model, dataset)    
    
    conf = conf.transpose()
    print 'test accuracy: %2.2f' % (100-err)
    print 'confusion matrix (cols true):'
    pp_array(100*conf/np.sum(conf, axis=0))

    # acc = file_misclass_error_topx(model, dataset, 2)
    # print 'test accuracy: %2.2f' % acc


