import argparse
import scikits.audiolab as audiolab
import numpy as np
from theano import function
from theano import tensor as T
from pylearn2.utils import serial
from audio_dataset import AudioDataset
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import pylearn2.config.yaml_parse as yaml_parse
import pdb

# min_y f(y)
# s.t. y >= 0 and ||y-x|| < e

# z = max(0, y^k - mu.f'(y^k))
# y^k+1 = P(z)

# P(z) = min_u ||u-z|| s.t. {u | ||u-x|| < e }
# Lagrangian(u,l) = L(u,l) = ||u-z|| + l(||u-x|| - e)
# dL/du = u-z + l(u-x) = 0
# u = (I+l)^-1 (z + l.x) = (1/(1+l)) (z + l.x)

# KKT:
# ||u-x|| = e
# ||(1/(1+l))(z + l.x) - x|| = e
# ||(1/(1+l))z + ((l/(1+l))-1)x|| = e
# ||(1/(1+l))z - (1/(1+l))x|| = e
# (1/(1+l))||z-x|| = e
# l = ||z-x||/e - 1

# find adversarial examples
def generate_adversary(model, X0, label, epsilon=.25):
    '''
    model - dnn model
    X0 - an example that the model classifies correctly
    label - an incorrect label
    '''

    # Computation of gradients using Theano
    X = model.get_input_space().make_theano_batch()
    label_vec = model.get_output_space().make_theano_batch()
    #label_vec = T.vector('label_vec')
    cost  = model.cost(label_vec, model.fprop(X))
    dCost = T.grad(cost, X) 
    f = function([X, label_vec], dCost)

    # convert integer label into one-hot vector
    n_classes = model.get_output_space().dim
    
    #one_hot = np.zeros(n_classes, dtype=np.float32)
    #one_hot[label] = 1
    
    n_examples = X0.shape[0]
    one_hot = np.zeros((n_examples, n_classes), dtype=np.float32)
    one_hot[:,label] = 1

    # compute gradient
    delta = f(X0, one_hot)
    X_adv = X0 - epsilon * n_examples * delta

    return X_adv * (X_adv>0)

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

    # window = np.hanning(nfft) # must use same window as compute_fft
    # x = np.zeros( X.shape[0]*(nhop+1) )
    # win_sum = np.zeros( X.shape[0]*(nhop+1) )

    # for i, frame in enumerate(X):
    #     sup = i*nhop + np.arange(nfft)
    #     x[sup] += np.real(np.fft.ifft(frame)) * window
    #     win_sum[sup] += window**2
    
    # return x/(win_sum + 1e-12)

    window = np.hanning(nfft) # must use same window as compute_fft
    x = np.zeros( X.shape[0]*(nhop+1) )
    win_sum = np.zeros( X.shape[0]*(nhop+1) )

    for i, frame in enumerate(X):
        sup = i*nhop + np.arange(nfft)
        x[sup] += np.real(np.fft.ifft(frame))
        win_sum[sup] += window # ensure perfect reconstruction
    
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
    model = serial.load(args.model)
    input_space = model.get_input_space()
    batch = input_space.make_theano_batch()
    fprop = function([batch], model.fprop(batch))

    # test_file = 'blues/blues.00060.wav'
    # dataset = yaml_parse.load(model.dataset_yaml_src)
    # offset, nframes, label, target = dataset.file_index[test_file]
    # X0 = dataset.X[offset:offset+nframes,:]
    # Y0 = fprop(X0)

    # compute fft of data
    nfft = 2*(input_space.dim-1)
    nhop = nfft//2
    x, fs, fmt = audiolab.wavread(args.test_file)
    Mag, Phs = compute_fft(x, nfft, nhop)

    # check classification
    # target = np.argmax(np.sum(y, axis=0))

    # attempt to generate nearby adversarial example that is classified as...
    # X_ad = []
    # for i, frame in enumerate(Mag[:,:input_space.dim]):
    #     print i
    #     X_ad.append( generate_adversary(model, frame.reshape((1,input_space.dim)), 1, .25) )
    # X_ad = np.vstack(X_ad)

    prediction = np.argmax(np.sum(fprop(Mag[:,:513]), axis=0))
    print 'Predicted label on original file: ', prediction

    mu = 0.025
    X_adv = generate_adversary(model, Mag[:,:input_space.dim], args.label, mu)
    for i in xrange(10):
        X_adv = generate_adversary(model, X_adv, args.label, mu)

    # test advesary
    prediction = np.argmax(np.sum(fprop(X_adv), axis=0))
    print 'Predicted label on adversarial example (before re-synthesis):', prediction

    # reconstruct time-domain sound
    x_adv = overlap_add( np.hstack((X_adv, X_adv[:,-2:-nfft/2-1:-1])) * np.exp(1j*Phs), nfft, nhop)
    #x_adv = griffin_lim( np.hstack((X_adv, X_adv[:,-2:-nfft/2-1:-1])) * np.exp(1j*Phs), 10, nfft, nhop)
    audiolab.wavwrite(x_adv, '/tmp/adversary.wav', fs, fmt)

    Mag2, Phs2 = compute_fft(x_adv, nfft, nhop)
    prediction = np.argmax(np.sum(fprop(Mag2[:,:513]), axis=0))
    print 'Predicted label on adversarial example (after re-synthesis): ', prediction
    # # find a 'correctly' classified frame
    # for i, (x, y) in enumerate(zip(X0, Y0)):
    #     print i
    #     if np.argmax(y)==target:
    #         print 'Found correctly classified frame (confidence: {})'.format(y[target])
    #         break

    # decoy = np.random.choice( np.setdiff1d(dataset.targets, [target]) )
    # print 'Attempting to find adversarial example with decoy label {} (true label should be {})'.format(decoy, target)
    # X_adv = generate_adversary(model, x.reshape((1,513)), decoy, epsilon=0.5)

    # #print 'Testing if adversarial example is mis-classified'
    # y = fprop(X_adv)[0]
    # prediction = np.argmax(y)
    # confidence = y[prediction]
    # if prediction != target:
    #     print 'Adversarial example mis-classified as {} (should be {}). Confidence in prediction is {}'.format(prediction, target, confidence)
    # else:
    #     print 'Unsuccessful at generating an adversary, confidence in prediction is: {}'.format(confidence)

