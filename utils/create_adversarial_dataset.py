import os, sys, re, csv, cPickle, argparse
import scikits.audiolab as audiolab
from sklearn.externals import joblib
import numpy as np
import theano
from theano import tensor as T
from pylearn2.utils import serial
from audio_dataset import AudioDataset
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import pylearn2.config.yaml_parse as yaml_parse
import pdb

def find_adversary(model, X0, label, mu=.1, epsilon=.25, maxits=10, stop_thresh=0.5):
    '''
    Solves:

    y* = argmin_y f(y; label) 
    s.t. y >= 0 and ||y-X0|| < e

    where f(y) is the cost associated the network associates with the pair (y,label)

    This can be solved using the projected gradient method:

    min_y f(y)
    s.t. y >= 0 and ||y-X0|| < e

    z = max(0, y^k - mu.f'(y^k))
    y^k+1 = P(z)

    P(z) = min_u ||u-z|| s.t. {u | ||u-X0|| < e }
    Lagrangian(u,l) = L(u,l) = ||u-z|| + nu*(||u-X0|| - e)
    dL/du = u-z + nu*(u-X0) = 0
    u = (1+nu)^-1 (z + nu*X0)

    KKT:
    ||u-x|| = e
    ||(1/(1+nu))(z + nu*x) - x|| = e
    ||(1/(1+nu))z + ((nu/(1+nu))-1)x|| = e
    ||(1/(1+nu))z - (1/(1+nu))x|| = e
    (1/(1+nu))||z-x|| = e
    nu = max(0,||z-x||/e - 1)

    function inputs:

    model - pylearn2 dnn model (implements fprop, cost)
    X0 - an example that the model classifies correctly
    label - an incorrect label
    '''
    # convert integer label into one-hot vector
    n_classes, n_examples = model.get_output_space().dim, X0.shape[0]     
    one_hot               = np.zeros((n_examples, n_classes), dtype=np.float32)
    one_hot[:,label]      = 1

    # Set-up gradient computation w/ Theano
    in_batch  = model.get_input_space().make_theano_batch()
    out_batch = model.get_output_space().make_theano_batch()
    cost      = model.cost(out_batch, model.fprop(in_batch))
    dCost     = T.grad(cost, in_batch)
    grad      = theano.function([in_batch, out_batch], dCost)
    fprop     = theano.function([in_batch], model.fprop(in_batch))

    # projected gradient:
    last_pred = 0
    Y = np.copy(X0)
    for i in xrange(maxits):        

        # gradient step
        Z = Y - mu * n_examples * grad(Y, one_hot)

        # non-negative projection
        Z = Z * (Z>=0)
        
        # maximum allowable signal-to-noise projection
        nu = np.linalg.norm(Z-X0)/n_examples/epsilon - 1 # lagrange multiplier
        nu = nu * (nu>=0)
        Y  = (Z + nu*X0) / (1+nu)

        # stopping condition
        pred = np.sum(fprop(Y), axis=0)
        pred /= np.sum(pred)

        print 'iteration: {}, pred[label]: {}, nu: {}'.format(i, pred[label], nu)
        if pred[label] > stop_thresh:
            break
        elif pred[label] < last_pred:
            break
        last_pred = pred[label]

    return Y

def file_misclass_error_printf(model, dataset, save_file, mode='all_same', label=0, snr=30, label_list=None):
    """
    Function to compute the file-level classification error by classifying
    individual frames and then voting for the class with highest cumulative probability
    """
    n_classes  = len(dataset.targets)
    feat_space = model.get_input_space()

    X     = feat_space.make_theano_batch()
    Y     = model.fprop(X)
    fprop = theano.function([X],Y)
    
    n_examples   = len(dataset.file_list)
    target_space = VectorSpace(dim=n_classes)
    data_specs   = (CompositeSpace((feat_space, target_space)), ("songlevel-features", "targets"))     
    iterator     = dataset.iterator(mode='sequential', batch_size=1, data_specs=data_specs)

    with open(save_file, 'w') as fname:
        csvwriter = csv.writer(fname, delimiter='\t')
        for i,el in enumerate(iterator):

            # display progress indicator
            sys.stdout.write('Progress: %2.0f%%\r' % (100*i/float(n_examples)))
            sys.stdout.flush()
        
            fft_data = np.array(el[0], dtype=np.float32)
            epsilon  = np.linalg.norm(fft_data)/fft_data.shape[0]/10**(snr/20)

            if mode == 'all_same':
                target = label
            elif mode == 'perfect':
                target = el[1]
            elif mode == 'random':
                target = np.random.randint(n_classes)

            fft_data = find_adversary(model, fft_data, target, mu=.1, epsilon=epsilon, maxits=100, stop_thresh=0.95)

            frame_labels = np.argmax(fprop(fft_data), axis=1)
            hist         = np.bincount(frame_labels, minlength=n_classes)
            
            if label_list: # use-string labels
                vote_label   = label_list[np.argmax(hist)] # most used label
                true_label   = dataset.label_list[el[1]]#np.argmax(el[1])
            else: # use numeric labels
                vote_label   = np.argmax(hist) # most used label
                true_label   = el[1] #np.argmax(el[1])

            csvwriter.writerow([dataset.file_list[i], true_label, vote_label])            
    print ''


if __name__ == '__main__':
    '''
    Variants:
    1) Label all excerpts the same (e.g., all blues)
    2) Perfect classification
    3) Random classification
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    description='''Script to find/test adversarial examples with a dnn''')
    parser.add_argument('--dnn_model', help='dnn model to use for features')
    #parser.add_argument('--model', help='model trained on dnn features (e.g. random forest)')
    
    # three variants
    parser.add_argument('--mode', help='either all_same, perfect, or random')
    parser.add_argument('--label', type=int, help='label to minimize loss on (only used in all_same mode)')
    
    parser.add_argument('--save_file', help='txt file to save results in')
    args = parser.parse_args()

    assert args.mode in ['all_same', 'perfect', 'random'] 
    if args.label is None:
        args.label = 0        

    model = serial.load(args.dnn_model)
    p = re.compile(r"which_set.*'(train)'")
    dataset_yaml = p.sub("which_set: 'test'", model.dataset_yaml_src)
    testset = yaml_parse.load(dataset_yaml)

    file_misclass_error_printf(model, testset, args.save_file, args.mode, args.label, snr=30, label_list=None)