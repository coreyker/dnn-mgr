import os, sys, re, csv, cPickle, argparse
import scikits.audiolab as audiolab
from utils.read_mp3 import read_mp3
from sklearn.externals import joblib
import numpy as np
import theano
from theano import tensor as T
from pylearn2.utils import serial
from audio_dataset import AudioDataset
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import pylearn2.config.yaml_parse as yaml_parse
from test_adversary import winfunc, compute_fft, overlap_add, griffin_lim_proj, find_adversary, aggregate_features
import pdb


def file_misclass_error_printf(dnn_model, root_dir, dataset, save_file, mode='all_same', label=0, snr=30, aux_model=None, aux_save_file=None, which_layers=None, save_adversary_audio=None):
    """
    Function to compute the file-level classification error by classifying
    individual frames and then voting for the class with highest cumulative probability
    """
    n_classes  = len(dataset.targets)    

    X     = dnn_model.get_input_space().make_theano_batch()
    Y     = dnn_model.fprop(X)
    fprop = theano.function([X],Y)
    
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

            if 1: # re-read audio (seems to be bug when reading from h5)
                if f.endswith('.wav'):
                    read_fun = audiolab.wavread             
                elif f.endswith('.au'):
                    read_fun = audiolab.auread
                elif f.endswith('.mp3'):
                    read_fun = read_mp3

                x,_,_ = read_fun(root_dir + el[2])
                Mag, Phs = compute_fft(x)
                Mag = Mag[:,:513]
                Phs = Phs[:,:513]

            X_adv, P_adv = find_adversary(
                model=dnn_model, 
                X0=Mag, 
                label=target, 
                P0=np.hstack((Phs, -Phs[:,-2:-dataset.nfft/2-1:-1])), 
                mu=.1, 
                epsilon=epsilon, 
                maxits=100, 
                stop_thresh=0.9, 
                griffin_lim=True)
            
            if save_adversary_audio: 
                
                nfft  = 2*(X_adv.shape[1]-1)
                nhop  = nfft//2      
                x_adv = overlap_add(np.hstack((X_adv, X_adv[:,-2:-nfft//2-1:-1])) * np.exp(1j*P_adv), nfft, nhop)
                audiolab.wavwrite(x_adv, os.path.join(save_adversary_audio, el[2]), 22050, 'pcm16')

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
    parser.add_argument('--root_dir', help='dataset directory')

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
        root_dir=args.root_dir,
        dataset=testset,         
        save_file=args.dnn_save_file, 
        mode=args.mode, 
        label=args.label, 
        snr=15., 
        aux_model=aux_model, 
        aux_save_file=args.aux_save_file, 
        which_layers=args.which_layers,
        save_adversary_audio=args.save_adversary_audio)
