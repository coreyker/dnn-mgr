import os, sys, re
import warnings
import numpy as np
import tables
import theano
import cPickle
from scikits import audiolab, samplerate
from utils.read_mp3 import read_mp3

import pdb

def collect_audio(root_directory, label_list):
    '''
    Find all audio files in the given directory and subdirectories. 
    Associate each file with a label:

    1. If the file resides in a directory whos name is in the label_list, apply that label
    2. If the file's name contains a label from the label list, apply that label
    
    1 takes precidence over 2.

    root_directory: base directory to start search
    label_list: list of categorical (alphanumeric) labels
    '''
    pwd = os.getcwd()
    os.chdir(root_directory) # is this bad? (chdir to get relative path names later on)
    label_list = [label.lower() for label in label_list] # make sure labels are lowercase
    
    file_dict = dict.fromkeys(label_list)
    for key in file_dict: file_dict[key]=list()

    for root, dirs, files in os.walk(root_directory):
        for filename in files:
            if filename.endswith(('.wav', '.au', '.mp3')):
                # audio file found, check for label
                labelled = False
                dir_label = os.path.split(root)[-1].lower()

                # check directory name first:
                for label in label_list:
                    match = re.search(label, root, re.IGNORECASE)
                    if match:
                        file_dict[match.group().lower()].append(os.path.relpath(os.path.join(root, filename), '.'))
                        labelled = True
                        break

                # check filename if now label found
                if not labelled:
                    for label in label_list:
                        match = re.search(label, filename, re.IGNORECASE)
                        if match:
                            file_dict[match.group().lower()].append(os.path.relpath(os.path.join(root, filename), '.'))
                            labelled = True
                            break

                if not labelled:
                    warnings.warn('Found audio file %s, but could not determine its label (please check that label is in filename or path)' % filename)
    os.chdir(pwd)

    for key in label_list:
        file_dict[key] = sorted(file_dict[key])
        print 'Found %d audio files with label %s' % (len(file_dict[key]), key)

    return file_dict

def make_hdf5(hdf5_save_name, label_list, root_directory='.', nfft=1024, nhop=512, fs=22050, seglen=30):

    if os.path.exists(hdf5_save_name):
        warnings.warn('hdf5 file {} already exists, new file will not be created'.format(hdf5_save_name))
        return

    file_dict  = collect_audio(root_directory, label_list)  

    # hdf5 setup
    hdf5_file  = tables.open_file(hdf5_save_name, mode = "w")
    data_node  = hdf5_file.create_group(hdf5_file.root, "Data", "Data")
    data_atom  = tables.Float32Atom() if theano.config.floatX == 'float32' else tables.Float64Atom()
    data_atom_complex = tables.ComplexAtom(8) if theano.config.floatX == 'float32' else tables.ComplexAtom(16)

    # data nodes
    hdf5_file.create_earray(data_node, 'X', atom=data_atom_complex, shape=(0,nfft/2+1), title="features")
    hdf5_file.create_earray(data_node, 'y', atom=data_atom, shape=(0,len(label_list)), title="targets")

    targets = range(len(label_list))
    window = np.hanning(nfft)

    file_index = {}
    offset = 0
    for target, key in zip(targets, label_list):
        print 'Processing %s' % key

        for f in file_dict[key.lower()]:
            
            if f.endswith('.wav'):
                read_fun = audiolab.wavread             
            elif f.endswith('.au'):
                read_fun = audiolab.auread
            elif f.endswith('.mp3'):
                read_fun = read_mp3
            
            # read audio
            audio_data, fstmp, _ = read_fun(os.path.join(root_directory, f))
            
            # make mono
            if len(audio_data.shape) != 1: 
                audio_data = np.sum(audio_data, axis=1)/2.
            
            # work with only first seglen seconds
            audio_data = audio_data[:fstmp*seglen] 

            # resample audio data
            if fstmp != fs:
                audio_data = samplerate.resample(audio_data, fs/float(fstmp), 'sinc_best')
            
            # compute dft
            nframes  = (len(audio_data)-nfft)//nhop
            fft_data = np.zeros((nframes, nfft))

            for i in xrange(nframes):
                sup = i*nhop + np.arange(nfft)
                fft_data[i,:] = audio_data[sup] * window
            
            fft_data = np.fft.fft(fft_data)

            # write dft frames to hdf5 file
            data_node.X.append(fft_data[:, :nfft/2+1]) # keeping phase too            

            # write target values to hdf5 file
            one_hot = np.zeros((nframes, len(label_list)))
            one_hot[:,target] = 1
            data_node.y.append(one_hot)

            # keep file-level info
            file_index[f] = (offset, nframes, key.lower(), target)
            offset += nframes

            hdf5_file.flush()

    # write file_index and dft parameters to hdf5 file
    param_node = hdf5_file.create_group(hdf5_file.root, "Param", "Param")
    param_atom = tables.ObjectAtom()
    
    # save dataset metadata
    hdf5_file.create_vlarray(param_node, 'file_index', atom=param_atom, title='file_index')
    param_node.file_index.append(file_index)
    
    hdf5_file.create_vlarray(param_node, 'file_dict', atom=param_atom, title='file_dict')
    param_node.file_dict.append(file_dict)

    hdf5_file.create_vlarray(param_node, 'fft', atom=param_atom, title='fft')
    param_node.fft.append({'nfft':nfft, 'nhop':nhop, 'window':window})

    hdf5_file.create_vlarray(param_node, 'label_list', atom=param_atom, title='label_list')
    param_node.label_list.append(label_list)
    
    hdf5_file.create_vlarray(param_node, 'targets', atom=param_atom, title='targets')
    param_node.targets.append(targets)

    hdf5_file.close()
    print '' # newline

def create_stratified_partition(hdf5, partition_save_prefix, train_prop=0.5, valid_prop=0.25, test_prop=0.25, tframes=1, compute_std=True, compute_pca=False):
    
    nfolds = int(np.reciprocal(test_prop))

    if np.linalg.norm(np.sum((train_prop, valid_prop, test_prop))-1)>1e-6:
        raise ValueError('train_prop + valid_prop + test_prop must add up to 1')    
    
    if np.linalg.norm(nfolds - np.reciprocal(test_prop))>1e-6:
        raise ValueError('Increase precision of test_prop')

    # extract metadata from dataset
    hdf5_file = tables.open_file(hdf5, mode='r')
    param     = hdf5_file.get_node('/', 'Param')    
    file_dict = param.file_dict[0]
    
    train_list = [[] for i in xrange(nfolds)]
    valid_list = [[] for i in xrange(nfolds)]
    test_list  = [[] for i in xrange(nfolds)]

    rng = np.random.RandomState(111)
    for key, files in file_dict.iteritems(): # for all files that share a given label
        nfiles = len(files)     
        ntest  = nfiles // nfolds
        ntrain = int(nfiles * train_prop)
        nvalid = nfiles - ntest - ntrain
        
        perm = rng.permutation(nfiles)      
        for fold in range(nfolds):
            sup         = fold*ntest + np.arange(ntest)
            test_index  = perm[sup]                     
            rest_index  = np.setdiff1d(perm, test_index)
            train_index = rest_index[:ntrain]
            valid_index = rest_index[ntrain:]
            
            train_list[fold].append([files[i] for i in train_index])
            valid_list[fold].append([files[i] for i in valid_index])
            test_list[fold].append([files[i] for i in test_index])

    # flatten lists
    for fold in xrange(nfolds):
        train_list[fold] = sum(train_list[fold],[])
        valid_list[fold] = sum(valid_list[fold],[])
        test_list[fold]  = sum(test_list[fold],[])

    for fold, (train, valid, test) in enumerate(zip(train_list, valid_list, test_list)):
        partition_save_name = os.path.splitext(partition_save_prefix)[0] + '-fold-%d_of_%d.pkl' % (fold+1, nfolds)
        
        if os.path.exists(partition_save_name):
            warnings.warn('partition file {} already exists, new file will not be created'.format(partition_save_name))
            continue
        else:       
            create_partition(hdf5, partition_save_name, train, valid, test, tframes, compute_std, compute_pca)
            print 'Created stratified partition %s' % partition_save_name
    
    hdf5_file.close()

def create_partiton_from_files(hdf5, partition_save_name, train_file, valid_file, test_file, tframes=1, compute_std=True, compute_pca=False):
    
    with open(train_file) as f: 
        train_list = [line.strip() for line in f.readlines()]

    if valid_file:
        with open(valid_file) as f: 
            valid_list = [line.strip() for line in f.readlines()]
    else:
        valid_list = None

    if test_file:
        with open(test_file) as f:  
            test_list = [line.strip() for line in f.readlines()]
    else:
        test_list = None

    create_partition(hdf5, partition_save_name, train_list, valid_list, test_list, tframes, compute_std, compute_pca)

def create_partition(hdf5, partition_save_name, train_list, valid_list=None, test_list=None, tframes=1, compute_std=True, compute_pca=False):
    
    if os.path.exists(partition_save_name):
        warnings.warn('partition file %s already exists, new file will not be created' % partition_save_name)
        return

    if not valid_list:
        valid_list = train_list
    if not test_list:
        test_list = train_list

    hdf5_file = tables.open_file(hdf5, mode='r')
    data   = hdf5_file.get_node('/', 'Data')
    nfeats = data.X.shape[1]
    param  = hdf5_file.get_node('/', 'Param')
    file_index = param.file_index[0]    

    train_support = []
    thop = 1.
    for f in train_list:
        offset, nframes, key, target = file_index[f]
        sup = np.arange(0,nframes-tframes,np.int(tframes/thop)) # hardcoded for now (!!must match with audio_dataset2d songlevel iterator!!)
        train_support.append(offset + sup) 
    train_support = np.hstack(train_support)

    valid_support = []
    for f in valid_list:
        offset, nframes, key, target = file_index[f]
        sup = np.arange(0,nframes-tframes,np.int(tframes/thop)) # hardcoded for now (!!must match with audio_dataset2d songlevel iterator!!)
        valid_support.append(offset + sup) 
    valid_support = np.hstack(valid_support)

    test_support = []
    for f in test_list:
        offset, nframes, key, target = file_index[f]
        sup = np.arange(0,nframes-tframes,np.int(tframes/thop)) # hardcoded for now (!!must match with audio_dataset2d songlevel iterator!!)
        test_support.append(offset + sup)  
    test_support = np.hstack(test_support)

    # compute mean and std for training set only
    class_means = None
    class_vars = None

    if compute_std:      
        
        nclasses = len(param.targets[0])
        sum_x = np.zeros((nclasses, tframes, nfeats), dtype=np.float32)
        sum_x2 = np.zeros((nclasses, tframes, nfeats), dtype=np.float32)

        nsamples = np.zeros(nclasses)
        for f in train_list:
            offset, nframes, key, target = file_index[f]
            sup = offset + np.arange(0,nframes-tframes,np.int(tframes/thop))
            
            for i in sup:                                
                fft_frame = np.abs(data.X[i:i+tframes,:])
                sum_x[target] += fft_frame
                sum_x2[target] += fft_frame**2
                nsamples[target] += 1

        class_means = sum_x / nsamples.reshape((nclasses,1,1))
        class_vars = (sum_x2 - sum_x**2 / nsamples.reshape((nclasses,1,1))) / (nsamples.reshape((nclasses,1,1))-1)

        mean = np.mean(class_means, axis=0)
        var = np.var(class_vars, axis=0)

        # nsamples = len(train_support)*tframes
        # sum_x  = np.zeros(nfeats, dtype=np.float32)
        # sum_x2 = np.zeros(nfeats, dtype=np.float32)  
        
        # for n,i in enumerate(train_support):
        #     sys.stdout.write('\rComputing mean and variance of training set: %2.2f%%' % (n*tframes/float(nsamples)*100))
        #     sys.stdout.flush()
        #     for j in xrange(tframes):                       
                
        #         fft_frame = np.abs(data.X[i+j,:])
        #         sum_x  += fft_frame
        #         sum_x2 += fft_frame**2
        # print ''

        # mean = sum_x / nsamples
        # var  = (sum_x2 - sum_x**2/nsamples)/(nsamples-1)
    else:
        mean = np.zeros(nfeats)
        var  = np.ones(nfeats)

    # compute PCA whitening matrix
    if compute_pca:
        XX = 0
        tmp_support = train_support[::3] # speed-up
        nsamples = len(tmp_support)*tframes
        for n,i in enumerate(tmp_support):
            sys.stdout.write('\rComputing PCA matrix: %2.2f%%' % (n*tframes/float(nsamples)*100))
            sys.stdout.flush()
            for j in xrange(tframes):                       
                
                fft_frame = np.abs(data.X[i+j,:])
                X = np.reshape(fft_frame - mean, (len(fft_frame), 1))
                XX += X.dot(X.T)
        print ''
        XX /= nsamples

        U,S,V = np.linalg.svd(XX)
    else:
        S = np.eye(nfeats)
        U = np.eye(nfeats)

    config = {
        'hdf5' : hdf5,
        'test' :  test_support, 
        'train' : train_support, 
        'valid' : valid_support,
        'train_files' : train_list,
        'valid_files' : valid_list,
        'test_files' : test_list,
        'class_means' : class_means,
        'class_vars' : class_vars,
        'mean' :  mean,
        'var' :   var,
        'tframes' : tframes,
        'U' : U,
        'S' : S
        }

    # pickle config     
    with open(partition_save_name, 'w') as f:
        cPickle.dump(config, f, protocol=2)
    
    hdf5_file.close()

if __name__=='__main__':
    
    import sys, argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description='''Script to prepare an audio dataset. Example usage:
        python prepare_dataset2.py ~/Datasets/tzanetakis_genre ~/Datasets/tzanetakis_genre/label_list.txt --train_prop 0.5 --valid_prop 0.25 --test_prop 0.25
        python prepare_dataset2.py ~/Datasets/tzanetakis_genre ~/Datasets/tzanetakis_genre/label_list.txt --train train_filtered2.txt --valid valid_filtered2.txt --test test_filtered2.txt
        ''')

    parser.add_argument('datadir', help='Path to dataset')
    parser.add_argument('labels', help="A CSV or newline separated list of the dataset's categorical labels")
    parser.add_argument('--hdf5', help='Name of hdf5 file to use')
    parser.add_argument('--nfft', help='FFT length to use when making hdf5 dataset',type=int)
    parser.add_argument('--nhop', help='Hop size to use when making hdf5 dataset',type=int)

    #parser.add_argument('--user-defined-partition', action='store_true', help='Use the --train, --valid, --')
    parser.add_argument('--train', help='path to newline seperated list of training files')
    parser.add_argument('--valid', help='path to newline seperated list of validation files')
    parser.add_argument('--test', help='path to newline seperated list of testing files')

    #parser.add_argument('--use-stratified-cv', action='store_true', help='Automatically generate partitions using stratified cross-validation')
    parser.add_argument('--train_prop', type=float)
    parser.add_argument('--valid_prop', type=float)
    parser.add_argument('--test_prop', type=float)

    parser.add_argument('--partition_name')
    parser.add_argument('--tframes', type=int)
    parser.add_argument('--compute_pca', action='store_true')
    parser.add_argument('--compute_std', action='store_true')

    args = parser.parse_args()
    
    # check validity of arugments
    # if len(sys.argv) < 3:
    #     parser.error('must specify either --train,--valid,--test files or --train_prop, --valid_prop, --test_prop')

    if (args.train is not None or args.valid is not None or args.test is not None) \
    and (args.train_prop is not None or args.valid_prop is not None or args.test_prop is not None):
        parser.error('either specify user supplied files with --train, --valid, --test, OR specify or use --train_prop, --valid_prop, --test_prop to automatically generate stratified partitions')

    if (args.train is None and args.valid is not None and args.test is not None):
        parser.error('if any of the flags --valid, --test are specified, then --train must be specified too')

    if (args.train_prop is not None or args.valid_prop is not None or args.test_prop is not None) \
    and (args.train_prop is None or args.valid_prop is None or args.test_prop is None):
        parser.error('if any of the flags --train_prop, --valid_prop, --test_prop are specified, then they all must be specified')

    # substitute defaults for missing values
    if args.hdf5 is None:
        dataset_name = os.path.split(os.path.abspath(args.datadir))[-1] + '.h5'
        args.hdf5 = os.path.join(args.datadir, dataset_name)
    
    if args.partition_name is None:
        dataset_name = os.path.split(os.path.abspath(args.datadir))[-1] + '.pkl'
        args.partition_name = os.path.join(args.datadir, dataset_name)

    if args.tframes is None:
        args.tframes = 1
    if args.nfft is None:
        args.nfft = 1024
    if args.nhop is None:
        args.nhop = 512

    with open(args.labels) as f:
        lines = f.readlines()
        if len(lines)==1: # assume comma separated, single line
            label_list = lines[0].replace(' ','').split(',')
        else:
            label_list = [l.split()[0] for l in lines]

        print 'Using labels:', label_list

    # create partitions
    print 'Preparing hdf5 file'
    make_hdf5(hdf5_save_name=args.hdf5, 
        label_list=label_list, 
        root_directory=args.datadir,
        nfft=args.nfft,
        nhop=args.nhop)

    if args.train is not None:
        'Print creating partition %s from files %s, %s, %s' % (args.partition_name, args.train, args.valid, args.test)
        create_partiton_from_files(hdf5=args.hdf5, 
            partition_save_name=args.partition_name, 
            train_file=args.train, 
            valid_file=args.valid, 
            test_file=args.test, 
            tframes=args.tframes, 
            compute_std=args.compute_std, 
            compute_pca=args.compute_pca)

    elif args.train_prop is not None:
        'Print creating stratified partitions'
        create_stratified_partition(hdf5=args.hdf5, 
            partition_save_prefix=args.partition_name, 
            train_prop=args.train_prop, 
            valid_prop=args.valid_prop, 
            test_prop=args.test_prop, 
            tframes=args.tframes,
            compute_std=args.compute_std, 
            compute_pca=args.compute_pca)
