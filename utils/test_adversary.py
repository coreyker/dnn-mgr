import argparse
import scikits.audiolab as audiolab
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
    return Y

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
        x[sup] += np.real(np.fft.ifft(frame)) # *window
        win_sum[sup] += window #**2 # ensure perfect reconstruction
    
    return x/(win_sum + 1e-12)

# def griffin_lim(X, nits=4, nfft=1024, nhop=512):

#     X0 = np.abs(X)
#     x  = overlap_add(X, nfft, nhop)
#     for i in xrange(nits):
#         _, Phs = compute_fft(x, nfft, nhop)
#         X = X0 * np.exp(1j*Phs)
#         x = overlap_add(X0, nfft, nhop)
#     return x

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    description='''Script to find/test adversarial examples with a dnn''')
    parser.add_argument('model', help='model to use')
    parser.add_argument('test_file', help='file to test model on')
    parser.add_argument('label', help='target to aim for')

    args = parser.parse_args()

    # load model, fprop function
    model       = serial.load(args.model)
    input_space = model.get_input_space()
    batch       = input_space.make_theano_batch()
    fprop       = theano.function([batch], model.fprop(batch))

    # compute fft of data
    nfft = 2*(input_space.dim-1)
    nhop = nfft//2
    x, fs, fmt = audiolab.wavread(args.test_file)
    Mag, Phs = compute_fft(x, nfft, nhop)

    X0 = Mag[:,:input_space.dim]
    prediction = np.argmax(np.sum(fprop(X0), axis=0))
    print 'Predicted label on original file: ', prediction

    snr = 30
    epsilon = np.linalg.norm(X0)/X0.shape[0]/10**(snr/20)
    X_adv = find_adversary(model, X0, args.label, mu=.1, epsilon=epsilon, maxits=100, stop_thresh=0.5)

    # test advesary
    prediction = np.argmax(np.sum(fprop(X_adv), axis=0))
    print 'Predicted label on adversarial example (before re-synthesis):', prediction

    # reconstruct time-domain sound
    x_adv = overlap_add( np.hstack((X_adv, X_adv[:,-2:-nfft/2-1:-1])) * np.exp(1j*Phs), nfft, nhop)
    #x_adv = griffin_lim( np.hstack((X_adv, X_adv[:,-2:-nfft/2-1:-1])) * np.exp(1j*Phs), 10, nfft, nhop)
    audiolab.wavwrite(x_adv, '/tmp/adversary.wav', fs, fmt)

    Mag2, Phs2 = compute_fft(x_adv, nfft, nhop)
    prediction = np.argmax(np.sum(fprop(Mag2[:,:input_space.dim]), axis=0))
    print 'Predicted label on adversarial example (after re-synthesis): ', prediction

