import os, sys, re, csv, cPickle
import numpy as np
import scipy as sp
import scikits.audiolab as audiolab
import scikits.samplerate as samplerate
from sklearn.externals import joblib
import theano

from pylearn2.utils import serial
from audio_dataset import AudioDataset

from test_adversary import aggregate_features, compute_fft
import pdb

def file_misclass_error_printf(dnn_model, aux_model, which_layers, data_dir, file_list, filter_cutoff, dnn_save_file, aux_save_file):
    
    # closures
    def dnn_classify(X):
        batch = dnn_model.get_input_space().make_theano_batch()
        fprop = theano.function([batch], dnn_model.fprop(batch))
        prediction = np.argmax(np.sum(fprop(X), axis=0))
        return prediction

    def aux_classify(X):
        Xagg = aggregate_features(dnn_model, X, which_layers)
        prediction = np.argmax(np.bincount(np.array(aux_model.predict(Xagg), dtype='int')))
        return prediction

    # filter coeffs
    b,a = sp.signal.butter(4, filter_cutoff/(22050./2.))
    
    dnn_file = open(dnn_save_file, 'w')
    aux_file = open(aux_save_file, 'w')        
    label_list = {'blues':0, 'classical':1, 'country':2, 'disco':3, 'hiphop':4, 'jazz':5, 'metal':6, 'pop':7, 'reggae':8, 'rock':9}

    for i, fname in enumerate(file_list):
        print 'Processing file {} of {}'.format(i+1, len(file_list))
        true_label = label_list[fname.split('/')[0]]

        x,_,_ = audiolab.wavread(os.path.join(data_dir, fname))
        x     = sp.signal.lfilter(b,a,x)
        X,_   = compute_fft(x)      
        X     = np.array(X[:,:513], dtype=np.float32)

        dnn_pred = dnn_classify(X)        
        dnn_file.write('{fname}\t{true_label}\t{pred_label}\n'.format(
            fname=fname,
            true_label=true_label,
            pred_label=dnn_pred))
  
        aux_pred = aux_classify(X)
        aux_file.write('{fname}\t{true_label}\t{pred_label}\n'.format(
            fname=fname,
            true_label=true_label,
            pred_label=aux_pred))

    dnn_file.close()
    aux_file.close()

def pp_array(array): # pretty printing
    for row in array:
        print ['%04.1f' % el for el in row]

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description='')

    parser.add_argument('--dnn_model', help='Path to trained dnn model file')
    parser.add_argument('--aux_model', help='Path to trained aux model file')
    parser.add_argument('--data_dir', help='Adversarial dataset dir')
    parser.add_argument('--test_list', help='List of test files in dataset dir')
    parser.add_argument('--filter_cutoff', type=float, help='filter cutoff')
    parser.add_argument('--dnn_save_file', help='')
    parser.add_argument('--aux_save_file', help='')
    args = parser.parse_args()
    
    # get model
    dnn_model = serial.load(args.dnn_model)
    aux_model = joblib.load(args.aux_model)
    L = os.path.splitext(os.path.split(args.aux_model)[-1])[0].split('_L')[-1]
    if L=='All':
        which_layers = [1,2,3]
    else:
        which_layers = [int(L)]

    with open(args.test_list) as f: 
        file_list = [l.strip() for l in f.readlines()]

    file_misclass_error_printf(
        dnn_model = dnn_model, 
        aux_model = aux_model, 
        which_layers = which_layers, 
        data_dir = args.data_dir, 
        file_list = file_list, 
        filter_cutoff = args.filter_cutoff, 
        dnn_save_file = args.dnn_save_file, 
        aux_save_file = args.aux_save_file)
