import os, argparse
import scikits.audiolab as audiolab
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import numpy as np
import theano
from theano import tensor as T
from pylearn2.utils import serial
from audio_dataset import AudioDataset
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import pylearn2.config.yaml_parse as yaml_parse
from utils.read_mp3 import read_mp3
import pdb

def find_adversary(model, X0, label, fwd_xform=None, back_xform=None, P0=None, mu=.1, epsilon=.25, maxits=10, stop_thresh=0.5, griffin_lim=False):
    '''
    *** Assumes input is not standardized (i.e., operates on raw inputs)
    *** If model does not include a standardization layer then 
    *** fwd_xform and back_xform must be be specified, where
    *** fwd_xform(x) = (x-mean)/std  and back_xform(x) = x*std + mean

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

    if back_xform is None: back_xform = lambda X: X
    if fwd_xform is None: fwd_xform = lambda X: X

    # convert integer label into one-hot vector
    n_classes, n_examples = model.get_output_space().dim, X0.shape[0]
    nfft = 2*(X0.shape[1]-1)
    nhop = nfft//2

    # Set-up gradient computation w/ Theano    
    in_batch  = model.get_input_space().make_theano_batch()
    out_batch = model.get_output_space().make_theano_batch()

    cost      = model.cost(out_batch, model.fprop(in_batch))
    dCost     = T.grad(cost * n_examples, in_batch)

    grad_theano = theano.function([in_batch, out_batch], dCost)
    fprop_theano = theano.function([in_batch], model.fprop(in_batch))
    fcost_theano = theano.function([in_batch, out_batch], cost)

    input_space = model.get_input_space()
    if isinstance(input_space, Conv2DSpace):
        tframes, dim = input_space.shape
        view_converter = DefaultViewConverter((tframes, dim, 1))
    else:
        dim = input_space.dim        
        tframes = 1
        view_converter = None

    nframes = X0.shape[0]
    thop = 1.
    sup = np.arange(0,nframes-tframes+1, np.int(tframes/thop))

    if view_converter is not None:
        def grad(batch, labels):            
            
            data = np.vstack([np.reshape(batch[i:i+tframes, :],(tframes*dim,)) for i in sup])            
            data = fwd_xform(data)

            topo_view = grad_theano(view_converter.get_formatted_batch(data, input_space), labels)
            design_mat = view_converter.topo_view_to_design_mat(topo_view)
            design_mat = back_xform(design_mat)

            return np.vstack([np.reshape(r, (tframes, dim)) for r in design_mat])

        def fprop(batch):
            
            data = np.vstack([np.reshape(batch[i:i+tframes, :],(tframes*dim,)) for i in sup])
            data = fwd_xform(data)

            return fprop_theano(view_converter.get_formatted_batch(data, input_space))

        def fcost(batch, labels):
        
            data = np.vstack([np.reshape(batch[i:i+tframes, :],(tframes*dim,)) for i in sup])
            data = fwd_xform(data)

            return fcost_theano(view_converter.get_formatted_batch(data, input_space), labels)
    else:
        grad = grad_theano
        fprop = fprop_theano
        fcost = fcost_theano        
    
    one_hot = np.zeros((len(sup), n_classes), dtype=np.float32)
    one_hot[:,label] = 1

    X0 = X0[:len(sup)*tframes,:]
    if P0 is not None: P0 = P0[:len(sup)*tframes,:]

    # projected gradient:
    last_pred = 0
    Y = np.copy(X0)
    Y_old = np.copy(Y)
    t_old = 1

    print 'cost(X0,y): ', fcost(X0, one_hot)

    for i in xrange(maxits):        

        # gradient step        
        g = grad(Y, one_hot)
        Z = Y - mu  * g
        print 'cost(X{},y): {}'.format(i+1, fcost(Z, one_hot))

        # non-negative projection
        Z = Z * (Z>0)

        if griffin_lim:
            Z, P0 = griffin_lim_proj(np.hstack((Z, Z[:,-2:-nfft/2-1:-1])), P0, its=0)
        
        # maximum allowable signal-to-noise projection
        nu = np.linalg.norm(Z-X0)/n_examples/epsilon - 1 # lagrange multiplier
        nu = nu * (nu>=0)
        Y  = (Z + nu*X0) / (1+nu)

        # FISTA momentum
        t = .5 + np.sqrt(1+4*t_old**2)/2.
        alpha = (t_old - 1)/t
        Y += alpha * (Y - Y_old)
        Y_old = np.copy(Y)
        t_old = t
        
        # stopping condition
        pred = np.sum(fprop(Y), axis=0)
        pred /= np.sum(pred)

        #print 'iteration: {}, pred[label]: {}, nu: {}'.format(i, pred[label], nu)
        print 'iteration: {}, pred[label]: {}, nu: {}, snr: {}'.format(i, pred[label], nu, 20*np.log10(np.linalg.norm(X0)/np.linalg.norm(Y-X0)))

        if pred[label] > stop_thresh:
            break
        elif pred[label] < last_pred - 1e-4:
            pass#break
        last_pred = pred[label]

    return Y, P0

winfunc = lambda x: np.hanning(x)
def compute_fft(x, nfft=1024, nhop=512):
  
    window   = winfunc(nfft)
    nframes  = int((len(x)-nfft)//nhop + 1)
    fft_data = np.zeros((nframes, nfft))

    for i in xrange(nframes):
        sup = i*nhop + np.arange(nfft)
        fft_data[i,:] = x[sup] * window
    
    fft_data = np.fft.fft(fft_data)
    return tuple((np.array(np.abs(fft_data), dtype=np.float32), np.array(np.angle(fft_data), dtype=np.float32)))

def overlap_add(X, nfft=1024, nhop=512):

    window = winfunc(nfft) # must use same window as compute_fft
    L = X.shape[0]*nhop + (nfft-nhop)
    x = np.zeros(L)
    win_sum = np.zeros(L)

    for i, frame in enumerate(X):
        sup = i*nhop + np.arange(nfft)
        x[sup] += np.real(np.fft.ifft(frame)) * window
        win_sum[sup] += window **2 # ensure perfect reconstruction
    
    return x/(win_sum + 1e-12)

def griffin_lim_proj(Mag, Phs=None, its=4, nfft=1024, nhop=512):
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
    parser.add_argument('--dnn_model', help='dnn model to use for features')
    parser.add_argument('--aux_model', help='(auxilliary) model trained on dnn features')
    parser.add_argument('--test_file', help='file to test model on')
    parser.add_argument('--label', type=int, help='target to aim for')
    parser.add_argument('--filter', type=float, help='apply filter to adversary and retest')
    parser.add_argument('--out_path', help='location for saving adversary (name automatically generated)')

    args = parser.parse_args()

    # load model, fprop function
    dnn_model   = serial.load(args.dnn_model)
    input_space = dnn_model.get_input_space()
    batch       = input_space.make_theano_batch()
    fprop_theano = theano.function([batch], dnn_model.fprop(batch))

    # load audio file
    if args.test_file.endswith('.wav'):
        read_fun = audiolab.wavread             
    elif args.test_file.endswith('.au'):
        read_fun = audiolab.auread
    elif args.test_file.endswith('.mp3'):
        read_fun = read_mp3

    x, fs, fmt = read_fun(args.test_file)
    
    # limit to first 30 seconds
    seglen = 30
    x = x[:seglen*fs]
    
    # make sure format agrees with training data
    if len(x.shape)!=1:
        print 'making mono:'
        x = np.sum(x, axis=1)/2. # mono
    if fs != 22050:
        print 'resampling to 22050 hz:'
        import scikits.samplerate as samplerate
        x = samplerate.resample(x, 22050./fs, 'sinc_best')
        fs = 22050

    if isinstance(input_space, Conv2DSpace):
        tframes, dim = input_space.shape
        view_converter = DefaultViewConverter((tframes, dim, 1))
    else:
        dim = input_space.dim        
        tframes = 1
        view_converter = None

    nfft = 2*(dim-1)
    nhop = nfft//2
    nframes = (len(x)-nfft)/nhop
    x = x[:(nframes-1)*nhop + nfft] # truncate input to multiple of hopsize

    # format batches for 1d/2d nets
    thop = 1.
    sup  = np.arange(0,nframes-tframes+1, np.int(tframes/thop))     
    if view_converter:
        def fprop(batch):
            data = np.vstack([np.reshape(batch[i:i+tframes, :],(tframes*dim,)) for i in sup])
            fprop_theano(view_converter.get_formatted_batch(data, input_space))
    else:
        fprop = fprop_theano

    # comput fft of input file
    # smooth boundaries to prevent a click
    win = winfunc(2048)
    x[:1024]  *= win[:1024]
    x[-1024:] *= win[1024:]
    Mag, Phs = compute_fft(x, nfft, nhop)

    X0 = Mag[:len(sup)*tframes,:dim]
    P0 = Phs[:len(sup)*tframes,:]

    prediction = np.argmax(np.sum(fprop(X0), axis=0))
    print 'Predicted label on original file: ', prediction

    snr = 15.
    epsilon = np.linalg.norm(X0)/X0.shape[0]/10**(snr/20.)

    X_adv, P_adv = find_adversary(model=dnn_model, 
        X0=X0,
        label=args.label, 
        P0=P0, 
        mu=0.1, 
        epsilon=epsilon, 
        maxits=100, 
        stop_thresh=0.9, 
        griffin_lim=True)

    # test advesary
    p1 = np.argmax(np.sum(fprop(X_adv), axis=0))
    print 'Predicted label on adversarial example (before re-synthesis):', p1

    # reconstruct time-domain sound
    x_adv = overlap_add( np.hstack((X_adv, X_adv[:,-2:-nfft/2-1:-1])) * np.exp(1j*P_adv))

    Mag2, Phs2 = compute_fft(x_adv, nfft, nhop)
    p2 = np.argmax(np.sum(fprop(Mag2[:len(sup)*tframes,:dim]), axis=0))
    print 'Predicted label on adversarial example (after re-synthesis): ', p2

    if args.filter:
        import scipy as sp
        b,a = sp.signal.butter(4, args.filter/(fs/2.))
        x_filt = sp.signal.lfilter(b,a,x_adv)

        Mag_filt,_ = compute_fft(x_filt, nfft, nhop)
        pf = np.argmax(np.sum(fprop(Mag_filt[:,:input_space.dim]), axis=0))
        print 'Predicted label on adversarial example (after filter): ', pf

    if args.aux_model: # now try with classifier trained on dnn features
        aux_model = joblib.load(args.aux_model)
        L = os.path.splitext(os.path.split(args.aux_model)[-1])[0].split('_L')[-1]
        if L=='All':
            which_layers = [1,2,3]
        else:
            which_layers = [int(L)]

        X_adv_agg = aggregate_features(dnn_model, X_adv, which_layers)
        p3 = np.argmax(np.bincount(np.array(aux_model.predict(X_adv_agg), dtype='int')))
        print 'Predicted label on adversarial example (classifier trained on aggregated features from last layer of dnn): ', p3


    if args.out_path:        
        out_snr   = 20*np.log10(np.linalg.norm(x[nfft:-nfft]) / np.linalg.norm(x[nfft:-nfft]-x_adv[nfft:-nfft]))
        label_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        out_label1 = label_list[p2]
        out_file1 = os.path.join(args.out_path, 'dnn',
            '{fname}.{label}_adversary.{snr}dB.dnn.wav'.format(fname=os.path.splitext(os.path.split(args.test_file)[-1])[0],
            label=out_label1,
            snr=int(out_snr+.5)))
        audiolab.wavwrite(x_adv, out_file1, fs, 'pcm16')

        if args.aux_model:
            out_label2 = label_list[p3]
            out_file2 = os.path.join(args.out_path, 'rf', 
                '{fname}.{label}_adversary.{snr}dB.rf.wav'.format(fname=os.path.splitext(os.path.split(args.test_file)[-1])[0],
                label=out_label2,
                snr=int(out_snr+.5)))
            audiolab.wavwrite(x_adv, out_file2, fs, 'pcm16')

    if 0:
        ## Time-domain waveforms
        ## ------------------------------------------------------------------------
        plt.ion()
        N = 512
        sup = np.arange(N)

        X_metal, P_metal = find_adversary(dnn_model, X0, 6, Phs, mu=.1, epsilon=epsilon, maxits=200, stop_thresh=0.9)        
        assert(np.argmax(fprop(X_metal[N:N+1,:]))==6)
        x_metal=overlap_add( np.hstack((X_metal, X_metal[:,-2:-nfft/2-1:-1])) * np.exp(1j*P_metal))
        
        L = min(len(x), len(x_metal))
        n = x[:L] - x_metal[:L]

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
        X_metal, P_metal = find_adversary(dnn_model, X0, 6, Phs, mu=.1, epsilon=epsilon, maxits=200, stop_thresh=0.9)        
        assert(np.argmax(fprop(X_metal[N:N+1,:]))==6)

        X_classical, P_classical = find_adversary(dnn_model, X0, 1, Phs, mu=.1, epsilon=epsilon, maxits=200, stop_thresh=0.9)
        assert(np.argmax(fprop(X_classical[N:N+1,:]))==1)

        X_disco, P_disco = find_adversary(dnn_model, X0, 3, Phs, mu=.1, epsilon=epsilon, maxits=200, stop_thresh=0.9)
        assert(np.argmax(fprop(X_disco)[N:N+1,:])==3)


        plt.figure()
        plt.gcf().set_tight_layout(True)
        x_range = np.arange(513)/513.*22.050
        ylim = (-60,40)
        plt.subplot(3,1,1)    
        plt.xlabel('(a)')
        plt.ylabel('dB')

        plt.plot(x_range, 20*np.log10(X0[N,:]), color=(.4,.6,1,0.8), linewidth=2)
        plt.axis('tight')
        plt.ylim(ylim)
        
        plt.plot(x_range, 20*np.log10(X_metal[N,:]), color=(0,0,0,1), linewidth=1)
        plt.axis('tight')
        plt.ylim(ylim)
        
        plt.plot(x_range, 20*np.log10(np.abs(X0[N,:]-X_metal[N,:])), '-', color=(1,0.6,0.1,0.6), linewidth=2)
        plt.axis('tight')
        plt.ylim(ylim)
        
        plt.subplot(3,1,2)
        plt.xlabel('(b)')
        plt.ylabel('dB')

        plt.plot(x_range, 20*np.log10(X0[N,:]), color=(.4,.6,1,0.8), linewidth=2)
        plt.axis('tight')
        plt.ylim(ylim)
        
        plt.plot(x_range, 20*np.log10(X_classical[N,:]), color=(0,0,0,1), linewidth=1)
        plt.axis('tight')
        plt.ylim(ylim)    
        
        plt.plot(x_range, 20*np.log10(np.abs(X0[N,:]-X_classical[N,:])), '-', color=(1,0.6,0.1,0.6), linewidth=2)
        plt.axis('tight')
        plt.ylim(ylim)
        
        plt.subplot(3,1,3)
        plt.xlabel('Frequency (kHz) \n (c)')
        plt.ylabel('dB')

        plt.plot(x_range, 20*np.log10(X0[N,:]), color=(.4,.6,1,0.8), linewidth=2)
        plt.axis('tight')
        plt.ylim(ylim)

        plt.plot(x_range, 20*np.log10(X_disco[N,:]), color=(0,0,0,1), linewidth=1)
        plt.axis('tight')
        plt.ylim(ylim)
        
        plt.plot(x_range, 20*np.log10(np.abs(X0[N,:]-X_disco[N,:])), '-', color=(1,0.6,0.1,0.6), linewidth=2)
        plt.axis('tight')
        plt.ylim(ylim)

        plt.savefig(os.path.splitext(out_file)[0] + '.spectrum.pdf', format='pdf')
