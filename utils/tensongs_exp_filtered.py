import os, argparse
import scikits.audiolab as audiolab
import scikits.samplerate as samplerate
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import numpy as np
import scipy as sp
import glob
import theano
from theano import tensor as T
from pylearn2.utils import serial
from audio_dataset import AudioDataset
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import pylearn2.config.yaml_parse as yaml_parse

from test_adversary import winfunc, compute_fft, overlap_add, griffin_lim_proj, find_adversary, aggregate_features

import pdb

# def find_adversary(model, X0, label, P0=None, mu=.1, epsilon=.25, maxits=10, stop_thresh=0.5, griffin_lim=False):
#     '''
#     Solves:

#     y* = argmin_y f(y; label) 
#     s.t. y >= 0 and ||y-X0|| < e

#     where f(y) is the cost associated the network associates with the pair (y,label)

#     This can be solved using the projected gradient method:

#     min_y f(y)
#     s.t. y >= 0 and ||y-X0|| < e

#     z = max(0, y^k - mu.f'(y^k))
#     y^k+1 = P(z)

#     P(z) = min_u ||u-z|| s.t. {u | ||u-X0|| < e }
#     Lagrangian(u,l) = L(u,l) = ||u-z|| + nu*(||u-X0|| - e)
#     dL/du = u-z + nu*(u-X0) = 0
#     u = (1+nu)^-1 (z + nu*X0)

#     KKT:
#     ||u-x|| = e
#     ||(1/(1+nu))(z + nu*x) - x|| = e
#     ||(1/(1+nu))z + ((nu/(1+nu))-1)x|| = e
#     ||(1/(1+nu))z - (1/(1+nu))x|| = e
#     (1/(1+nu))||z-x|| = e
#     nu = max(0,||z-x||/e - 1)

#     function inputs:

#     model - pylearn2 dnn model (implements fprop, cost)
#     X0 - an example that the model classifies correctly
#     label - an incorrect label
#     '''
#     # convert integer label into one-hot vector
#     n_classes, n_examples = model.get_output_space().dim, X0.shape[0]
#     one_hot               = np.zeros((n_examples, n_classes), dtype=np.float32)
#     one_hot[:,label]      = 1

#     # Set-up gradient computation w/ Theano
#     in_batch  = model.get_input_space().make_theano_batch()
#     out_batch = model.get_output_space().make_theano_batch()
#     cost      = model.cost(out_batch, model.fprop(in_batch))
#     dCost     = T.grad(cost, in_batch)
#     grad      = theano.function([in_batch, out_batch], dCost)
#     fprop     = theano.function([in_batch], model.fprop(in_batch))

#     # projected gradient:
#     last_pred = 0
#     #Y = np.array(np.random.rand(*X0.shape), dtype=np.float32) 
#     Y = np.copy(X0)
#     Y_old = np.copy(Y)
#     t_old = 1
#     for i in xrange(maxits):        

#         # gradient step
#         Z = Y - mu * n_examples * grad(Y, one_hot)

#         # non-negative projection
#         Z = Z * (Z>0)

#         if griffin_lim:
#             Z, P0 = griffin_lim_proj(np.hstack((Z, Z[:,-2:-nfft/2-1:-1])), P0, its=0)
        
#         # maximum allowable signal-to-noise projection
#         nu = np.linalg.norm((Z-X0))/n_examples/epsilon - 1 # lagrange multiplier
#         nu = nu * (nu>=0)
#         Y  = (Z + nu*X0) / (1+nu)
        
#         # FISTA momentum
#         t = .5 + np.sqrt(1+4*t_old**2)/2.
#         alpha = (t_old - 1)/t
#         Y += alpha * (Y - Y_old)
#         Y_old = np.copy(Y)
#         t_old = t

#         # stopping condition
#         pred = np.sum(fprop(Y), axis=0)
#         pred /= np.sum(pred)

#         #print 'iteration: {}, pred[label]: {}, nu: {}'.format(i, pred[label], nu)
#         print 'iteration: {}, pred[label]: {}, nu: {}, snr: {}'.format(i, pred[label], nu, 20*np.log10(np.linalg.norm(X0)/np.linalg.norm(Y-X0)))

#         if pred[label] > stop_thresh:
#             break
#         elif pred[label] < last_pred + 1e-4:
#             break
#         last_pred = pred[label]

#     return Y, P0

# winfunc = lambda x: np.hanning(x)
# def compute_fft(x, nfft=1024, nhop=512):
  
#     window   = winfunc(nfft)
#     nframes  = int((len(x)-nfft)//nhop + 1)
#     fft_data = np.zeros((nframes, nfft))

#     for i in xrange(nframes):
#         sup = i*nhop + np.arange(nfft)
#         fft_data[i,:] = x[sup] * window
    
#     fft_data = np.fft.fft(fft_data)
#     return tuple((np.array(np.abs(fft_data), dtype=np.float32), np.array(np.angle(fft_data), dtype=np.float32)))

# def overlap_add(X, nfft=1024, nhop=512):

#     window = winfunc(nfft) # must use same window as compute_fft
#     L = X.shape[0]*nhop + (nfft-nhop)
#     x = np.zeros(L)
#     win_sum = np.zeros(L)

#     for i, frame in enumerate(X):
#         sup = i*nhop + np.arange(nfft)
#         x[sup] += np.real(np.fft.ifft(frame)) * window
#         win_sum[sup] += window **2 # ensure perfect reconstruction
    
#     return x/(win_sum + 1e-12)

# def griffin_lim_proj(Mag, Phs=None, its=4, nfft=1024, nhop=512):
#     if Phs is None:
#         Phs = np.pi * np.random.randn(*Mag.shape)

#     x = overlap_add(Mag * np.exp(1j*Phs), nfft, nhop)
#     for i in xrange(its):
#         _, Phs = compute_fft(x, nfft, nhop)
#         x = overlap_add(Mag * np.exp(1j*Phs), nfft, nhop)

#     Mag, Phs = compute_fft(x, nfft, nhop)
#     return np.array(Mag[:,:nfft//2+1], dtype=np.float32), Phs


# def aggregate_features(model, X, which_layers=[3], win_size=200, step=100):
#     assert np.max(which_layers) < len(model.layers)

#     n_classes, n_examples = model.get_output_space().dim, X.shape[0] 
#     in_batch              = model.get_input_space().make_theano_batch()    
#     fprop                 = theano.function([in_batch], model.fprop(in_batch, return_all=True))
#     output_data           = fprop(X)
#     feats                 = np.hstack([output_data[i] for i in which_layers])

#     agg_feat = []
#     for i in xrange(0, feats.shape[0]-win_size, step):
#         chunk = feats[i:i+win_size,:]
#         agg_feat.append(np.hstack((np.mean(chunk, axis=0), np.std(chunk, axis=0))))
        
#     return np.vstack(agg_feat)

def stripf(f):
    fname = os.path.split(f)[-1]
    return os.path.splitext(fname)[0]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    description='')
    parser.add_argument('--dnn_model', help='dnn model to use for features')
    parser.add_argument('--aux_model', help='(auxilliary) model trained on dnn features')
    parser.add_argument('--in_path', help='file to test model on')
    parser.add_argument('--out_path', help='location for saving adversary (name automatically generated)')

    args = parser.parse_args()

    # tunable alg. parameters
    snr = 15.
    mu  = .05
    stop_thresh = .9
    maxits = 100
    cut_freq = 9000.

    label_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    targets     = range(len(label_list))

    # load dnn model, fprop function
    dnn_model   = serial.load(args.dnn_model)
    input_space = dnn_model.get_input_space()
    batch       = input_space.make_theano_batch()
    fprop       = theano.function([batch], dnn_model.fprop(batch))

    # load aux model
    aux_model = joblib.load(args.aux_model)
    L = os.path.splitext(os.path.split(args.aux_model)[-1])[0].split('_L')[-1]
    if L=='All':
        which_layers = [1,2,3]
    else:
        which_layers = [int(L)]

    # fft params
    nfft = 2*(input_space.dim-1)
    nhop = nfft//2
    win = winfunc(1024)
    
    # design lowpass filter.
    b,a = sp.signal.butter(4, cut_freq/(22050./2.))

    flist = glob.glob(args.in_path +'*.wav')

    dnn_file = open(os.path.join(args.out_path, stripf(args.dnn_model) + '.adversaries.txt'), 'w')
    dnn_file_filt = open(os.path.join(args.out_path, stripf(args.dnn_model) + '.adversaries.filtered.txt'), 'w')
    aux_file = open(os.path.join(args.out_path, stripf(args.aux_model) + '.adversaries.txt'), 'w')
    aux_file_filt = open(os.path.join(args.out_path, stripf(args.aux_model) + '.adversaries.filtered.txt'), 'w')

    for f in flist:
        fname = stripf(f)

        # load audio file
        x, fs, fmt = audiolab.wavread(f)
    
        # make sure format agrees with training data
        if len(x.shape)!=1:
            print 'making mono:'
            x = np.sum(x, axis=1)/2. # mono
        if fs != 22050:
            print 'resampling to 22050 hz:'
            x = samplerate.resample(x, 22050./fs, 'sinc_best')
            fs = 22050
        
        # truncate input to multiple of hopsize
        nframes = (len(x)-nfft)/nhop
        x = x[:(nframes-1)*nhop + nfft] 

        # smooth boundaries to prevent a click    
        x[:512]  *= win[:512]
        x[-512:] *= win[512:]

        # compute mag. spectra
        Mag, Phs = compute_fft(x, nfft, nhop)
        X0 = Mag[:,:input_space.dim]
            
        epsilon = np.linalg.norm(X0)/X0.shape[0]/10**(snr/20.)

        # write file name
        dnn_file.write('{}\t'.format(fname))
        dnn_file_filt.write('{}\t'.format(fname))
        aux_file.write('{}\t'.format(fname))
        aux_file_filt.write('{}\t'.format(fname))

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
            x_adv   = overlap_add( np.hstack((X_adv, X_adv[:,-2:-nfft/2-1:-1])) * np.exp(1j*P_adv))
            out_snr = 20*np.log10(np.linalg.norm(x[nfft:-nfft]) / np.linalg.norm(x[nfft:-nfft]-x_adv[nfft:-nfft]))

            # BEFORE FILTERING
            # ===========================================
            # dnn prediction
            pred = np.argmax(np.sum(fprop(X_adv), axis=0))
            if pred == t:
                dnn_file.write('{}\t'.format(int(out_snr+.5)))
            else:
                dnn_file.write('{}\t'.format('na'))

            # aux prediction
            X_adv_agg = aggregate_features(dnn_model, X_adv, which_layers)
            pred = np.argmax(np.bincount(np.array(aux_model.predict(X_adv_agg), dtype='int')))
            if pred == t:
                aux_file.write('{}\t'.format(int(out_snr+.5)))
            else:
                aux_file.write('{}\t'.format('na'))

            # filtered representation
            x_filt = sp.signal.lfilter(b,a,x_adv)
            Mag2, Phs2 = compute_fft(x_filt, nfft, nhop)
            X_adv_filt = Mag2[:,:input_space.dim]            

            # AFTER FILTERING
            # ==================================================
            # dnn prediction
            pred = np.argmax(np.sum(fprop(X_adv_filt), axis=0))
            if pred == t:
                dnn_file_filt.write('{}\t'.format('x'))
            else:
                dnn_file_filt.write('{}\t'.format('o'))

            # aux prediction
            X_adv_agg_filt = aggregate_features(dnn_model, X_adv_filt, which_layers)
            pred = np.argmax(np.bincount(np.array(aux_model.predict(X_adv_agg_filt), dtype='int')))
            if pred == t:
                aux_file_filt.write('{}\t'.format('x'))
            else:
                aux_file_filt.write('{}\t'.format('o'))

            # SAVE ADVERSARY FILES
            out_file = os.path.join(args.out_path,
            '{fname}.{label}.adversary.{snr}dB.wav'.format(
                fname=fname,
                label=label_list[t],
                snr=int(out_snr+.5)))
            audiolab.wavwrite(x_adv, out_file, fs, fmt)

            out_file2 = os.path.join(args.out_path,
            '{fname}.{label}.adversary.filtered.wav'.format(
                fname=fname,
                label=label_list[t]))
            audiolab.wavwrite(x_filt, out_file2, fs, fmt)

        dnn_file.write('\n'.format(fname))
        dnn_file_filt.write('\n'.format(fname))
        aux_file.write('\n'.format(fname))
        aux_file_filt.write('\n'.format(fname))

    dnn_file.close()
    dnn_file_filt.close()
    aux_file.close()
    aux_file_filt.close()
