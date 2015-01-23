import os, argparse
import scikits.audiolab as audiolab
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import numpy as np
import theano
from theano import tensor as T
from pylearn2.utils import serial
from audio_dataset import AudioDataset
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import pylearn2.config.yaml_parse as yaml_parse
import pdb

def find_adversary(model, X0, label, P0=None, mu=.1, epsilon=.25, maxits=10, stop_thresh=0.5):
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
    delta = 0
    for i in xrange(maxits):        

        # gradient step
        Z = Y - mu * n_examples * grad(Y, one_hot)

        # non-negative projection
        Z = Z * (Z>=0)

        if P0 is not None:
            Y, P0 = griffin_lim(np.hstack((Y, Y[:,-2:-nfft/2-1:-1])), P0, its=4)
        
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

    return Y, P0

def compute_fft(x, nfft=1024, nhop=512):
  
    window   = np.hanning(nfft)
    nframes  = (len(x)-nfft)//nhop
    fft_data = np.zeros((nframes, nfft))

    for i in xrange(nframes):
        sup = i*nhop + np.arange(nfft)
        fft_data[i,:] = x[sup] * window
    
    fft_data = np.fft.fft(fft_data)
    return tuple((np.array(np.abs(fft_data), dtype=np.float32), np.array(np.angle(fft_data), dtype=np.float32)))

def overlap_add(X, nfft=1024, nhop=512):

    window = np.hanning(nfft) # must use same window as compute_fft
    x = np.zeros( X.shape[0]*(nhop+1) )
    win_sum = np.zeros( X.shape[0]*(nhop+1) )

    for i, frame in enumerate(X):
        sup = i*nhop + np.arange(nfft)
        x[sup] += np.real(np.fft.ifft(frame)) * window
        win_sum[sup] += window **2 # ensure perfect reconstruction
    
    return x/(win_sum + 1e-12)

def griffin_lim(Mag, Phs=None, its=4, nfft=1024, nhop=512):
    if Phs is None:
        Phs = np.pi * np.random.randn(*Mag.shape)

    x = overlap_add(Mag * np.exp(1j*Phs), nfft, nhop)
    for i in xrange(its):
        _, Phs = compute_fft(x, nfft, nhop)
        x = overlap_add(Mag * np.exp(1j*Phs), nfft, nhop)

    Mag, Phs = compute_fft(x, nfft, nhop)
    return np.array(Mag[:,:nfft//2+1], dtype=np.float32), Phs


def aggregate_features(model, X, which_layers=[3], win_size=200, step=100):
    assert np.max(which_layers) < len(model.layers)

    n_classes, n_examples = model.get_output_space().dim, X.shape[0] 
    in_batch              = model.get_input_space().make_theano_batch()    
    fprop                 = theano.function([in_batch], model.fprop(in_batch, return_all=True))
    output_data           = fprop(X)
    feats                 = np.hstack([output_data[i] for i in which_layers])

    agg_feat = []
    for i in xrange(0, feats.shape[0]-win_size, step):
        chunk = feats[i:i+win_size,:]
        agg_feat.append(np.hstack((np.mean(chunk, axis=0), np.std(chunk, axis=0))))
        
    return np.vstack(agg_feat)

if __name__ == '__main__':
    '''
    Example call:
    python path/to/dnn-mgr/utils/test_adversary.py /path/to/dnn/S_50_RS.pkl /path/to/rf/S_50_RS_AF_LAll.pkl blues/blues.00001.wav 5
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    description='''Script to find/test adversarial examples with a dnn''')
    parser.add_argument('dnn_model', help='dnn model to use for features')
    parser.add_argument('model', help='model trained on dnn features')
    parser.add_argument('test_file', help='file to test model on')
    parser.add_argument('label', help='target to aim for')

    args = parser.parse_args()

    # load model, fprop function
    model       = serial.load(args.dnn_model)
    input_space = model.get_input_space()
    batch       = input_space.make_theano_batch()
    fprop       = theano.function([batch], model.fprop(batch))

    # compute fft of data
    nfft = 2*(input_space.dim-1)
    nhop = nfft//2
    x, fs, fmt = audiolab.wavread(args.test_file)
    s1 = np.copy(x)

    Mag, Phs = compute_fft(x, nfft, nhop)

    X0 = Mag[:,:input_space.dim]
    prediction = np.argmax(np.sum(fprop(X0), axis=0))
    print 'Predicted label on original file: ', prediction

    snr = 30
    epsilon = np.linalg.norm(X0)/X0.shape[0]/10**(snr/20)
    X_adv, P_adv = find_adversary(model, X0, args.label, Phs, mu=.01, epsilon=epsilon, maxits=100, stop_thresh=0.51)

    # test advesary
    prediction = np.argmax(np.sum(fprop(X_adv), axis=0))
    print 'Predicted label on adversarial example (before re-synthesis):', prediction

    # reconstruct time-domain sound
    x_adv = overlap_add( np.hstack((X_adv, X_adv[:,-2:-nfft/2-1:-1])) * np.exp(1j*P_adv))
    out_file = os.path.join('/tmp', os.path.splitext(os.path.split(args.test_file)[-1])[0] + '.adversary.wav')
    audiolab.wavwrite(x_adv, out_file, fs, fmt)

    Mag2, Phs2 = compute_fft(x_adv, nfft, nhop)
    prediction = np.argmax(np.sum(fprop(Mag2[:,:input_space.dim]), axis=0))
    print 'Predicted label on adversarial example (after re-synthesis): ', prediction

    # now try with classifier trained on last layer of dnn features
    clf = joblib.load(args.model)
    L   = os.path.splitext(os.path.split(args.model)[-1])[0].split('_L')[-1]
    if L=='All':
        which_layers = [1,2,3]
    else:
        which_layers = [int(L)]

    X_adv_agg = aggregate_features(model, X_adv, which_layers)
    prediction = np.argmax(np.bincount(np.array(clf.predict(X_adv_agg), dtype='int')))
    print 'Predicted label on adversarial example (classifier trained on aggregated features from last layer of dnn): ', prediction

    if 0:
        ## Time-domain waveforms
        ## ------------------------------------------------------------------------
        plt.ion()
        X_metal, P_metal = find_adversary(model, X0, 6, Phs, mu=.1, epsilon=epsilon, maxits=200, stop_thresh=0.95)        
        assert(np.argmax(fprop(X_metal[N:N+1,:]))==6)
        x_metal=overlap_add( np.hstack((X_metal, X_metal[:,-2:-nfft/2-1:-1])) * np.exp(1j*P_metal))
        
        L = min(len(x), len(x_metal))
        n = x[:L] - x_metal[:L]

        N = 512
        sup = np.arange(N)

        offsets = np.sort(np.random.randint(0, L-N, 6))
        for i,offset in enumerate(offsets):
            plt.subplot(6,1,i)
            plt.plot(sup, n[offset+sup],  '-', color=(1,0.6,0.1,1), linewidth=1) 
            plt.plot(sup, x[offset+sup], '-', color=(.4,.6,1,0.6), linewidth=6)
            plt.plot(sup, x_metal[offset+sup], '-', color=(0,0,0,1), linewidth=1)        
            plt.axis('tight')
            plt.axis('off')

        plt.savefig(os.path.splitext(out_file)[0] + '.pdf', format='pdf')
        
        ## Spectrum
        ## ------------------------------------------------------------------------    
        N = 50
        X_metal, P_metal = find_adversary(model, X0, 6, Phs, mu=.1, epsilon=epsilon, maxits=200, stop_thresh=0.95)        
        assert(np.argmax(fprop(X_metal[N:N+1,:]))==6)

        X_classical, P_classical = find_adversary(model, X0, 1, Phs, mu=.1, epsilon=epsilon, maxits=200, stop_thresh=0.95)
        assert(np.argmax(fprop(X_classical[N:N+1,:]))==1)

        X_disco, P_disco = find_adversary(model, X0, 3, Phs, mu=.1, epsilon=epsilon, maxits=200, stop_thresh=0.9)
        assert(np.argmax(fprop(X_disco)[N:N+1,:])==3)


        plt.figure()
        plt.gcf().set_tight_layout(True)

        ylim = (-60,40)
        plt.subplot(3,1,1)    
        plt.xlabel('(a)')
        plt.ylabel('dB')

        plt.plot(20*np.log10(X0[N,:]), color=(.4,.6,1,0.8), linewidth=2)
        plt.axis('tight')
        plt.ylim(ylim)
        
        plt.plot(20*np.log10(X_metal[N,:]), color=(0,0,0,1), linewidth=1)
        plt.axis('tight')
        plt.ylim(ylim)
        
        plt.plot(20*np.log10(np.abs(X0[N,:]-X_metal[N,:])), '-', color=(1,0.6,0.1,0.6), linewidth=2)
        plt.axis('tight')
        plt.ylim(ylim)
        
        plt.subplot(3,1,2)
        plt.xlabel('(b)')
        plt.ylabel('dB')

        plt.plot(20*np.log10(X0[N,:]), color=(.4,.6,1,0.8), linewidth=2)
        plt.axis('tight')
        plt.ylim(ylim)
        
        plt.plot(20*np.log10(X_classical[N,:]), color=(0,0,0,1), linewidth=1)
        plt.axis('tight')
        plt.ylim(ylim)    
        
        plt.plot(20*np.log10(np.abs(X0[N,:]-X_classical[N,:])), '-', color=(1,0.6,0.1,0.6), linewidth=2)
        plt.axis('tight')
        plt.ylim(ylim)
        
        plt.subplot(3,1,3)
        plt.xlabel('(c)')
        plt.ylabel('dB')

        plt.plot(20*np.log10(X0[N,:]), color=(.4,.6,1,0.8), linewidth=2)
        plt.axis('tight')
        plt.ylim(ylim)

        plt.plot(20*np.log10(X_disco[N,:]), color=(0,0,0,1), linewidth=1)
        plt.axis('tight')
        plt.ylim(ylim)
        
        plt.plot(20*np.log10(np.abs(X0[N,:]-X_disco[N,:])), '-', color=(1,0.6,0.1,0.6), linewidth=2)
        plt.axis('tight')
        plt.ylim(ylim)
