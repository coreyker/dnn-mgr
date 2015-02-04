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


def file_misclass_error_printf(dnn_model, dataset, save_file, mode='all_same', label=0, snr=30, aux_model=None, aux_save_file=None, which_layers=None, save_adversary_audio=None):
    """
    Function to compute the file-level classification error by classifying
    individual frames and then voting for the class with highest cumulative probability
    """
    n_classes  = len(dataset.targets)    

    X     = dnn_model.get_input_space().make_theano_batch()
    Y     = dnn_model.fprop(X)
    fprop = theano.function([X],Y)
    
    n_examples   = len(dataset.file_list)
    target_space = VectorSpace(dim=n_classes)
    feat_space   = VectorSpace(dim=dataset.nfft//2+1, dtype='complex64')
    data_specs   = (CompositeSpace((feat_space, target_space)), ("songlevel-features", "targets"))     
    iterator     = dataset.iterator(mode='sequential', batch_size=1, data_specs=data_specs)

    if aux_model:
        aux_fname = open(aux_save_file, 'w')
        aux_writer = csv.writer(aux_fname, delimiter='\t')

    with open(save_file, 'w') as fname:
        dnn_writer = csv.writer(fname, delimiter='\t')
        for i,el in enumerate(iterator):

            # display progress indicator
            'Progress: %2.0f%%\r' % (100*i/float(n_examples))
        
            Mag, Phs = np.abs(el[0], dtype=np.float32), np.angle(el[0])
            epsilon  = np.linalg.norm(Mag)/Mag.shape[0]/10**(snr/20.)

            if mode == 'all_same':
                target = label
            elif mode == 'perfect':
                target = el[1]
            elif mode == 'random':
                target = np.random.randint(n_classes)

            X_adv, P_adv = find_adversary(model=dnn_model, 
                X0=Mag, 
                label=target, 
                P0=np.hstack((Phs, -Phs[:,-2:-dataset.nfft/2-1:-1])), 
                mu=.05, 
                epsilon=epsilon, 
                maxits=50, 
                stop_thresh=0.9, 
                griffin_lim=griffin_lim)
            
            # if not griffin_lim:
            #     X_adv_valid, _ = griffin_lim_proj(np.hstack((X_adv, X_adv[:,-2:-nfft//2-1:-1])), P_adv, its=0)

            pdb.set_trace()
            if save_adversary_audio: 
                
                nfft  = 2*(X_adv.shape[1]-1)
                nhop  = nfft//2      
                x_adv = overlap_add(np.hstack((X_adv, X_adv[:,-2:-nfft//2-1:-1])) * np.exp(1j*P_adv), nfft, nhop)
                audiolab.wavewrite(x_adv, os.path.join(save_adversary_audio, el[2]), 22050, 'pcm16')

            frame_labels = np.argmax(fprop(X_adv), axis=1)
            hist         = np.bincount(frame_labels, minlength=n_classes)
            
            dnn_label    = np.argmax(hist) # most used label
            true_label   = el[1] #np.argmax(el[1])

            dnn_writer.writerow([dataset.file_list[i], true_label, dnn_label]) 

            print 'Mode: {}, True label: {}, DNN adversarial label: {}'.format(mode, true_label, dnn_label)
            if aux_model:
                fft_agg  = aggregate_features(dnn_model, X_adv, which_layers)
                aux_vote = np.argmax(np.bincount(np.array(aux_model.predict(fft_agg), dtype='int')))
                aux_writer.writerow([dataset.file_list[i], true_label, aux_vote]) 
                print 'AUX adversarial label: {}'.format(aux_vote)
    if aux_model:
        aux_fname.close()
    print ''

def find_adversary(model, X0, label, P0=None, mu=.1, epsilon=.25, maxits=10, stop_thresh=0.5, griffin_lim=False):
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
    nfft = 2*(X0.shape[1]-1)

    # Set-up gradient computation w/ Theano
    in_batch  = model.get_input_space().make_theano_batch()
    out_batch = model.get_output_space().make_theano_batch()
    cost      = model.cost(out_batch, model.fprop(in_batch))
    dCost     = T.grad(cost, in_batch)
    grad      = theano.function([in_batch, out_batch], dCost)
    fprop     = theano.function([in_batch], model.fprop(in_batch))

    # projected gradient:
    last_pred = 0
    #Y = np.array(np.random.rand(*X0.shape), dtype=np.float32) 
    Y = np.copy(X0)
    Y_old = np.copy(Y)
    t_old = 1
    for i in xrange(maxits):        

        # gradient step
        Z = Y - mu * n_examples * grad(Y, one_hot)

        # non-negative projection
        Z = Z * (Z>0)

        if griffin_lim:
            Z, P0 = griffin_lim_proj(np.hstack((Z, Z[:,-2:-nfft//2-1:-1])), P0, its=0)
        
        # maximum allowable signal-to-noise projection
        nu = np.linalg.norm((Z-X0))/n_examples/epsilon - 1 # lagrange multiplier
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
        elif pred[label] < last_pred + 1e-4:
            break
        last_pred = pred[label]

    return Y, P0

winfunc = lambda x: np.hanning(x)
def compute_fft(x, nfft=1024, nhop=512):
  
    window   = winfunc(nfft)
    nframes  = (len(x)-nfft)//nhop
    fft_data = np.zeros((nframes, nfft))

    for i in xrange(nframes):
        sup = i*nhop + np.arange(nfft)
        fft_data[i,:] = x[sup] * window
    
    fft_data = np.fft.fft(fft_data)
    return tuple((np.array(np.abs(fft_data), dtype=np.float32), np.array(np.angle(fft_data), dtype=np.float32)))

def overlap_add(X, nfft=1024, nhop=512):

    window = winfunc(nfft) # must use same window as compute_fft
    x = np.zeros( X.shape[0]*(nhop+1) )
    win_sum = np.zeros( X.shape[0]*(nhop+1) )

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
    Variants:
    1) Label all excerpts the same (e.g., all blues)
    2) Perfect classification
    3) Random classification
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    description='''Script to find/test adversarial examples with a dnn''')
    parser.add_argument('--dnn_model', help='dnn model to use for features')
    parser.add_argument('--aux_model', help='(optional) auxiliary model trained on dnn features (e.g. random forest)')
    parser.add_argument('--which_layers', nargs='*', type=int, help='(optional) layer(s) from dnn to be passed to auxiliary model')

    # three variants
    parser.add_argument('--mode', help='either all_same, perfect, or random')
    parser.add_argument('--label', type=int, help='label to minimize loss on (only used in all_same mode)')
    
    parser.add_argument('--dnn_save_file', help='txt file to save results in')
    parser.add_argument('--aux_save_file', help='txt file to save results in')
    parser.add_argument('--save_adversary_audio', help='path to save adversaries')

    args = parser.parse_args()

    assert args.mode in ['all_same', 'perfect', 'random'] 
    if args.mode == 'all_same' and not args.label:
        parser.error('--label x must be specified together with all_same mode')
    if args.aux_model and not args.which_layers:
        parser.error('--which_layers x1 x2 ... must be specified together with aux_model')
    if args.aux_model and not args.aux_save_file:
        parser.error('--aux_save_file x must be specified together with --aux_model')      

    dnn_model = serial.load(args.dnn_model)
    p = re.compile(r"which_set.*'(train)'")
    dataset_yaml = p.sub("which_set: 'test'", dnn_model.dataset_yaml_src)
    testset = yaml_parse.load(dataset_yaml)

    if args.aux_model:
        aux_model = joblib.load(args.aux_model)
    else:
        aux_model = None

    file_misclass_error_printf(
        dnn_model=dnn_model, 
        dataset=testset, 
        save_file=args.dnn_save_file, 
        mode=args.mode, 
        label=args.label, 
        snr=15., 
        aux_model=aux_model, 
        aux_save_file=args.aux_save_file, 
        which_layers=args.which_layers,
        save_adversary_audio=args.save_adversary_audio)
