import sys, re, cPickle
import numpy as np
import theano

from pylearn2.utils import serial
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import pylearn2.config.yaml_parse as yaml_parse
import audio_dataset

import pdb
# def frame_misclass_error(model, dataset):
#     """
#     Function to compute the frame-level classification error by classifying
#     individual frames and then voting for the class with highest cumulative probability
#     """

#     X = model.get_input_space().make_theano_batch()
#     Y = model.fprop( X )
#     fprop = theano.function([X],Y)

#     n_classes  = dataset.raw.y.shape[1]
#     confusion  = np.zeros((n_classes, n_classes))
#     n_examples = len(dataset.raw.support)
#     n_frames_per_file = dataset.raw.n_frames_per_file

#     batch_size = n_frames_per_file
#     data_specs = dataset.raw.get_data_specs()
#     iterator = dataset.iterator(mode='sequential', 
#         batch_size=batch_size, 
#         data_specs=data_specs
#         )

#     i=0
#     for el in iterator:

#         # display progress indicator
#         sys.stdout.write('Classify progress: %2.0f%%\r' % (100*i/float(n_examples)))
#         sys.stdout.flush()
    
#         fft_data    = np.array(el[0], dtype=np.float32)
#         vote_labels = np.argmax(fprop(fft_data), axis=1)
#         true_labels = np.argmax(el[1], axis=1)

#         for l,v in zip(true_labels, vote_labels):
#             confusion[l, v] += 1

#         i += batch_size

#     total_error = 100*(1 - np.sum(np.diag(confusion)) / np.sum(confusion))
#     print ''
#     return total_error, confusion

def frame_misclass_error(model, dataset):
    """
    Function to compute the frame-level classification error by classifying
    individual frames and then voting for the class with highest cumulative probability
    """

    n_classes  = len(dataset.targets)
    feat_space = model.get_input_space()

    X     = feat_space.make_theano_batch()
    Y     = model.fprop( X )
    fprop = theano.function([X],Y)
    
    confusion  = np.zeros((n_classes, n_classes))
    
    batch_size   = 2400
    n_examples   = len(dataset.support) // batch_size
    target_space = VectorSpace(dim=n_classes)
    data_specs   = (CompositeSpace((feat_space, target_space)), ("features", "targets"))     
    iterator     = dataset.iterator(mode='sequential', batch_size=batch_size, data_specs=data_specs)

    for i, el in enumerate(iterator):

        # display progress indicator
        sys.stdout.write('Classify progress: %2.0f%%\r' % (100*i/float(n_examples)))
        sys.stdout.flush()
    
        fft_data    = np.array(el[0], dtype=np.float32)
        vote_labels = np.argmax(fprop(fft_data), axis=1)
        true_labels = np.argmax(el[1], axis=1)

        for l,v in zip(true_labels, vote_labels):
            confusion[l, v] += 1

    total_error = 100*(1 - np.sum(np.diag(confusion)) / np.sum(confusion))
    print ''
    return total_error, confusion

def file_misclass_error(model, dataset):
    """
    Function to compute the file-level classification error by classifying
    individual frames and then voting for the class with highest cumulative probability
    """
    n_classes  = len(dataset.targets)
    feat_space = model.get_input_space()
    
    X     = feat_space.make_theano_batch()
    Y     = model.fprop( X )
    fprop = theano.function([X],Y)

    confusion  = np.zeros((n_classes, n_classes))
    n_examples = len(dataset.file_list)

    target_space = VectorSpace(dim=n_classes)

    data_specs   = (CompositeSpace((feat_space, target_space)), ("songlevel-features", "targets"))     
    iterator     = dataset.iterator(mode='sequential', batch_size=1, data_specs=data_specs)

    for i,el in enumerate(iterator):

        # display progress indicator
        sys.stdout.write('Classify progress: %2.0f%%\r' % (100*i/float(n_examples)))
        sys.stdout.flush()
    
        fft_data     = np.array(el[0], dtype=np.float32)
        frame_labels = np.argmax(fprop(fft_data), axis=1)
        hist         = np.bincount(frame_labels, minlength=n_classes)
        vote_label   = np.argmax(hist) # most used label
        #vote_label = np.argmax(np.sum(fprop(fft_data), axis=0))
        true_label = el[1] #np.argmax(el[1])
        confusion[true_label, vote_label] += 1

    total_error = 100*(1 - np.sum(np.diag(confusion)) / np.sum(confusion))
    print ''
    return total_error, confusion


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
    parser.add_argument('--which_set', help='train, test, or valid')
    args = parser.parse_args()
    
    
    # get model
    model = serial.load(args.model_file)  

    if args.testset: # dataset config passed in from command line
        print 'Using dataset passed in from command line:'
        with open(args.testset) as f: config = cPickle.load(f)
        dataset = audio_dataset2d.AudioDataset(config=config, which_set=args.which_set)

        # get model dataset for its labels...
        model_dataset = yaml_parse.load(model.dataset_yaml_src)
        label_list = model_dataset.label_list

    else: # get dataset from model's yaml_src
        print "Using dataset from model's yaml src"
        p = re.compile(r"which_set.*'(train)'")
        dataset_yaml = p.sub("which_set: '{}'".format(args.which_set), model.dataset_yaml_src)
        dataset = yaml_parse.load(dataset_yaml)
        
        label_list = dataset.label_list

    # test error
    #err, conf = frame_misclass_error(model, dataset)
    err, conf = file_misclass_error(model, dataset)
    
    conf = conf.transpose()
    print 'test accuracy: %2.2f' % (100-err)
    print 'confusion matrix (cols true):'
    pp_array(100*conf/np.sum(conf, axis=0))

    # acc = file_misclass_error_topx(model, dataset, 2)
    # print 'test accuracy: %2.2f' % acc


