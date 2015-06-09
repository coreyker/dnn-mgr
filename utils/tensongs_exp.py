import os, argparse
from scikits import audiolab, samplerate
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import numpy as np
import scipy as sp
import glob
import theano
from theano import tensor as T
from pylearn2.utils import serial
from audio_dataset import AudioDataset
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import pylearn2.config.yaml_parse as yaml_parse
from utils.read_mp3 import read_mp3

from test_adversary import winfunc, compute_fft, overlap_add, griffin_lim_proj, find_adversary, aggregate_features

import pdb

def stripf(f):
    fname = os.path.split(f)[-1]
    return os.path.splitext(fname)[0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    description='')
    parser.add_argument('--dnn_model', help='dnn model to use for features')
    parser.add_argument('--aux_model', help='(auxilliary) model trained on dnn features')
    parser.add_argument('--labels', help='(auxilliary) model trained on dnn features')
    parser.add_argument('--in_path', help='directory with files to test model on')
    parser.add_argument('--out_path', help='location for saving adversary (name automatically generated)')

    args = parser.parse_args()

    # tunable alg. parameters
    snr = 15.
    mu  = .1
    stop_thresh = .9
    maxits = 100

    with open(args.labels) as f:
        lines = f.readlines()
        if len(lines)==1: # assume comma separated, single line
            label_list = lines[0].replace(' ','').split(',')
        else:
            label_list = [l.split()[0] for l in lines]

    targets = range(len(label_list))

    # load dnn model, fprop function
    dnn_model    = serial.load(args.dnn_model)
    input_space  = dnn_model.get_input_space()
    batch        = input_space.make_theano_batch()
    fprop_theano = theano.function([batch], dnn_model.fprop(batch))

    if isinstance(input_space, Conv2DSpace):
        tframes, dim = input_space.shape
        view_converter = DefaultViewConverter((tframes, dim, 1))
    else:
        dim = input_space.dim        
        tframes = 1
        view_converter = None

    if view_converter:
        def fprop(batch):
            nframes = batch.shape[0]
            thop = 1.
            sup = np.arange(0,nframes-tframes+1, np.int(tframes/thop))
            data = np.vstack([np.reshape(batch[i:i+tframes, :],(tframes*dim,)) for i in sup])
            return fprop_theano(view_converter.get_formatted_batch(data, input_space))
    else:
        fprop = fprop_theano

    # load aux model
    if args.aux_model:
        aux_model = joblib.load(args.aux_model)
        L = os.path.splitext(os.path.split(args.aux_model)[-1])[0].split('_L')[-1]
        if L=='All':
            which_layers = [1,2,3]
        else:
            which_layers = [int(L)]
        aux_file = open(os.path.join(args.out_path, stripf(args.aux_model) + '.adversaries.txt'), 'w')

    dnn_file = open(os.path.join(args.out_path, stripf(args.dnn_model) + '.adversaries.txt'), 'w')

    # fft params
    nfft = 2*(dim-1)
    nhop = nfft//2
    win = winfunc(2048)
    
    flist = glob.glob(os.path.join(args.in_path, '*'))

    for f in flist:
        fname = stripf(f)

        if f.endswith('.wav'):
            read_fun = audiolab.wavread             
        elif f.endswith('.au'):
            read_fun = audiolab.auread
        elif f.endswith('.mp3'):
            read_fun = read_mp3
        else:
            continue

        x, fstmp, _ = read_fun(f)

        # make mono
        if len(x.shape) != 1: 
            x = np.sum(x, axis=1)/2.

        seglen=30
        x = x[:fstmp*seglen]

        fs = 22050
        if fstmp != fs:
            x = samplerate.resample(x, fs/float(fstmp), 'sinc_best')

        # compute mag. spectra
        Mag, Phs = compute_fft(x, nfft, nhop)
        X0 = Mag[:,:dim]
            
        epsilon = np.linalg.norm(X0)/X0.shape[0]/10**(snr/20.)

        # write file name
        dnn_file.write('{}\t'.format(fname))
        if args.aux_model:
            aux_file.write('{}\t'.format(fname))
 
        for t in targets:

            # search for adversary
            X_adv, P_adv = find_adversary(
                model=dnn_model, 
                X0=X0, 
                label=t, 
                P0=Phs, 
                mu=mu, 
                epsilon=epsilon, 
                maxits=maxits, 
                stop_thresh=stop_thresh, 
                griffin_lim=True)

            # get time-domain representation
            x_adv = overlap_add( np.hstack((X_adv, X_adv[:,-2:-nfft/2-1:-1])) * np.exp(1j*P_adv))
            
            minlen = min(len(x_adv), len(x))
            x_adv = x_adv[:minlen]
            x = x[:minlen] 
            out_snr = 20*np.log10(np.linalg.norm(x[nfft:-nfft]) / np.linalg.norm(x[nfft:-nfft]-x_adv[nfft:-nfft]))

           # dnn prediction
            pred = np.argmax(np.sum(fprop(X_adv), axis=0))
            if pred == t:
                dnn_file.write('{}\t'.format(int(out_snr+.5)))
            else:
                dnn_file.write('{}\t'.format('na'))

            # aux prediction
            if args.aux_model:
                X_adv_agg = aggregate_features(dnn_model, X_adv, which_layers)
                pred = np.argmax(np.bincount(np.array(aux_model.predict(X_adv_agg), dtype='int')))
                if pred == t:
                    aux_file.write('{}\t'.format(int(out_snr+.5)))
                else:
                    aux_file.write('{}\t'.format('na'))

            # SAVE ADVERSARY FILES
            out_file = os.path.join(args.out_path,
            '{fname}.{label}.adversary.{snr}dB.wav'.format(
                fname=fname,
                label=label_list[t],
                snr=int(out_snr+.5)))
            audiolab.wavwrite(x_adv, out_file, fs, fmt)

        dnn_file.write('\n'.format(fname))
        if args.aux_model:
            aux_file.write('\n'.format(fname))
    
    dnn_file.close()
    if args.aux_model:
        aux_file.close()
    