import os, sys, re, csv, cPickle, argparse
from scikits import audiolab, samplerate
from utils.read_mp3 import read_mp3
from sklearn.externals import joblib
import numpy as np
import theano
from theano import tensor as T
from pylearn2.utils import serial
from audio_dataset import AudioDataset, PreprocLayer
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
import pylearn2.config.yaml_parse as yaml_parse
from test_adversary import winfunc, compute_fft, overlap_add, griffin_lim_proj, find_adversary, aggregate_features
import pdb


def file_misclass_error_printf(dnn_model, root_dir, dataset, save_file, mode='all_same', label=0, snr=30, aux_model=None, aux_save_file=None, which_layers=None, save_adversary_audio=None, fwd_xform=None, back_xform=None):
    """
    Function to compute the file-level classification error by classifying
    individual frames and then voting for the class with highest cumulative probability
    """
    if fwd_xform is None: 
        print 'fwd_xform=None, using identity'
        fwd_xform = lambda X: X
    if back_xform is None: 
        print 'back_xform=None, using identity'
        back_xform = lambda X: X

    n_classes  = len(dataset.targets)    

    X     = dnn_model.get_input_space().make_theano_batch()
    Y     = dnn_model.fprop(X)
    fprop_theano = theano.function([X],Y)

    input_space = dnn_model.get_input_space()
    if isinstance(input_space, Conv2DSpace):
        tframes, dim = input_space.shape
        view_converter = DefaultViewConverter((tframes, dim, 1))
    else:
        dim = input_space.dim        
        tframes = 1
        view_converter = None

    if view_converter is not None:
        def fprop(batch):
            nframes = batch.shape[0]
            thop = 1.
            sup = np.arange(0,nframes-tframes+1, np.int(tframes/thop))
            
            data = np.vstack([np.reshape(batch[i:i+tframes, :],(tframes*dim,)) for i in sup])
            data = fwd_xform(data)
            
            return fprop_theano(view_converter.get_formatted_batch(data, input_space))

    else:
        fprop = fprop_theano

    n_examples   = len(dataset.file_list)
    target_space = dnn_model.get_output_space() #VectorSpace(dim=n_classes)
    feat_space   = dnn_model.get_input_space() #VectorSpace(dim=dataset.nfft//2+1, dtype='complex64')
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
            elif mode == 'all_wrong':
                cand = np.setdiff1d(np.arange(n_classes),np.array(el[1])) # remove ground truth label from set of options
                target = cand[np.random.randint(len(cand))]

            if 1: # re-read audio (seems to be bug when reading from h5)
                f = el[2]
                if f.endswith('.wav'):
                    read_fun = audiolab.wavread             
                elif f.endswith('.au'):
                    read_fun = audiolab.auread
                elif f.endswith('.mp3'):
                    read_fun = read_mp3

                x, fstmp, _ = read_fun(os.path.join(root_dir, f))

                # make mono
                if len(x.shape) != 1: 
                    x = np.sum(x, axis=1)/2.

                seglen=30
                x = x[:fstmp*seglen]

                fs = 22050
                if fstmp != fs:
                    x = samplerate.resample(x, fs/float(fstmp), 'sinc_best')

                Mag, Phs = compute_fft(x)
                Mag = Mag[:1200,:513]
                Phs = Phs[:1200,:513]
                epsilon = np.linalg.norm(Mag)/Mag.shape[0]/10**(snr/20.)
            else:
                raise ValueError("Check that song-level iterator is indeed returning 'raw data'") 

            X_adv, P_adv = find_adversary(
                model=dnn_model, 
                X0=Mag, 
                label=target,
                fwd_xform=fwd_xform,
                back_xform=back_xform,
                P0=np.hstack((Phs, -Phs[:,-2:-dataset.nfft/2-1:-1])), 
                mu=.15, 
                epsilon=epsilon, 
                maxits=10, 
                stop_thresh=0.9, 
                griffin_lim=True)
            
            if save_adversary_audio: 
                
                nfft  = 2*(X_adv.shape[1]-1)
                nhop  = nfft//2      
                x_adv = overlap_add(np.hstack((X_adv, X_adv[:,-2:-nfft//2-1:-1])) * np.exp(1j*P_adv), nfft, nhop)
                audiolab.wavwrite(x_adv, os.path.join(save_adversary_audio, el[2]), 22050, 'pcm16')

            #frame_labels = np.argmax(fprop(X_adv), axis=1)
            #hist         = np.bincount(frame_labels, minlength=n_classes)
            
            fpass = fprop(X_adv)
            conf = np.sum(fpass, axis=0) / float(fpass.shape[0])
            dnn_label = np.argmax(conf) #np.argmax(hist) # most used label
            true_label = el[1]

            # truncate to correct length
            ext = min(Mag.shape[0], X_adv.shape[0])
            Mag = Mag[:ext,:]
            X_adv = X_adv[:ext,:]

            X_diff = Mag-X_adv
            out_snr = 20*np.log10(np.linalg.norm(Mag)/np.linalg.norm(X_diff))
            
            dnn_writer.writerow([dataset.file_list[i], true_label, dnn_label, out_snr, conf[dnn_label]]) 

            print 'Mode:{}, True label:{}, Adv label:{}, Sel label:{}, Conf:{}, Out snr: {}'.format(mode, true_label, target, dnn_label, conf[dnn_label], out_snr)
            if aux_model:
                fft_agg  = aggregate_features(dnn_model, X_adv, which_layers)
                aux_vote = np.argmax(np.bincount(np.array(aux_model.predict(fft_agg), dtype='int')))
                aux_writer.writerow([dataset.file_list[i], true_label, aux_vote]) 
                print 'AUX adversarial label: {}'.format(aux_vote)
    if aux_model:
        aux_fname.close()
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
    parser.add_argument('--aux_model', help='(optional) auxiliary model trained on dnn features (e.g. random forest)')
    parser.add_argument('--which_layers', nargs='*', type=int, help='(optional) layer(s) from dnn to be passed to auxiliary model')

    # three variants
    parser.add_argument('--mode', help='either all_same, perfect, random, all_wrong')
    parser.add_argument('--label', type=int, help='label to minimize loss on (only used in all_same mode)')
    parser.add_argument('--root_dir', help='dataset directory')

    parser.add_argument('--dnn_save_file', help='txt file to save results in')
    parser.add_argument('--aux_save_file', help='txt file to save results in')
    parser.add_argument('--save_adversary_audio', help='path to save adversaries')

    args = parser.parse_args()

    assert args.mode in ['all_same', 'perfect', 'random', 'all_wrong'] 
    if args.mode == 'all_same' and not args.label:
        parser.error('--label x must be specified together with all_same mode')
    if args.aux_model and not args.which_layers:
        parser.error('--which_layers x1 x2 ... must be specified together with aux_model')
    if args.aux_model and not args.aux_save_file:
        parser.error('--aux_save_file x must be specified together with --aux_model')      

    dnn_model = serial.load(args.dnn_model)
    if isinstance(dnn_model.layers[0], PreprocLayer):
        print 'Preprocessing layer detected'
        fwd_xform = None
        back_xform = None
    else:
        print 'No preprocessing layer detected'
        trainset = yaml_parse.load(dnn_model.dataset_yaml_src)
        fwd_xform = lambda batch: (batch - trainset.mean) * trainset.istd * trainset.mask
        back_xform = lambda batch: (batch / trainset.istd + trainset.mean) * trainset.mask 
    
    p = re.compile(r"which_set.*'(train)'")
    dataset_yaml = p.sub("which_set: 'test'", dnn_model.dataset_yaml_src)
    testset = yaml_parse.load(dataset_yaml)

    if args.aux_model:
        aux_model = joblib.load(args.aux_model)
    else:
        aux_model = None

    file_misclass_error_printf(
        dnn_model=dnn_model, 
        root_dir=args.root_dir,
        dataset=testset,         
        save_file=args.dnn_save_file, 
        mode=args.mode, 
        label=args.label, 
        snr=15., 
        aux_model=aux_model, 
        aux_save_file=args.aux_save_file, 
        which_layers=args.which_layers,
        save_adversary_audio=args.save_adversary_audio,
        fwd_xform=fwd_xform,
        back_xform=back_xform)

