import os, sys, re, cPickle
import numpy as np
import theano

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from pylearn2.utils import serial
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import pylearn2.config.yaml_parse as yaml_parse

import pdb

def aggregate_features(model, dataset, which_layers=[2], win_size=200, step=100):
    assert np.max(which_layers) < len(model.layers)

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X, return_all=True)
    fprop = theano.function([X],Y)

    n_classes  = dataset.raw.y.shape[1]
    n_examples = len(dataset.raw.file_list)

    feat_space   = model.get_input_space()
    target_space = VectorSpace(dim=n_classes)

    data_specs = (CompositeSpace((feat_space, target_space)), ("songlevel-features", "targets"))     
    iterator = dataset.iterator(mode='sequential', batch_size=1, data_specs=data_specs)

    # compute feature representation, aggregrate frames
    X=[]; y=[]; Z=[];
    for n,el in enumerate(iterator):
        # display progress indicator
        sys.stdout.write('Aggregation progress: %2.0f%%\r' % (100*n/float(n_examples)))
        sys.stdout.flush()

        input_data  = np.array(el[0], dtype=np.float32)
        output_data = fprop(input_data)
        feats = np.hstack([output_data[i] for i in which_layers])
        
        Z.append(np.sum(output_data[-1], axis=0))

        # aggregate features
        mean=[]; std=[]
        for i in xrange(0, feats.shape[0]-win_size, step):
            chunk = feats[i:i+win_size,:]
            mean.append( np.mean(chunk, axis=0) )
            std.append( np.std(chunk, axis=0) )
        
        X.append( np.hstack((np.vstack(mean), np.vstack(std))) )

        labels = np.argmax(el[1], axis=1)
        true_label = labels[0]
        for entry in labels:
            assert entry == true_label # check for indexing prob

        y.append(true_label)

    print '' # newline
    return X, y, Z

def get_features(model, dataset, which_layers=[2]):
    assert np.max(which_layers) < len(model.layers)

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X, return_all=True)
    fprop = theano.function([X],Y)

    n_classes  = dataset.raw.y.shape[1]
    n_examples = len(dataset.raw.file_list)

    feat_space   = model.get_input_space()
    target_space = VectorSpace(dim=n_classes)

    data_specs = (CompositeSpace((feat_space, target_space)), ("songlevel-features", "targets"))     
    iterator = dataset.iterator(mode='sequential', batch_size=1, data_specs=data_specs)
    
    X=[]; y=[]; Z=[];
    for n,el in enumerate(iterator):
        # display progress indicator
        sys.stdout.write('Getting features: %2.0f%%\r' % (100*n/float(n_examples)))
        sys.stdout.flush()

        input_data  = np.array(el[0], dtype=np.float32)
        output_data = fprop(input_data)
        feats = np.hstack([output_data[i] for i in which_layers])

        Z.append(np.sum(output_data[-1], axis=0))
        X.append(feats)

        labels = np.argmax(el[1], axis=1)
        true_label = labels[0]
        for entry in labels:
            assert entry == true_label # check for indexing prob

        y.append(true_label)
    print ''
    return X, y, Z

def train_classifier(X_train, y_train, method='random_forest', verbose=2):
    assert method in ['random_forest', 'linear_svm']
    
    # train classifier
    if method=='random_forest':
        classifier = RandomForestClassifier(n_estimators=500, random_state=1234, verbose=verbose, n_jobs=2)
    else:
        classifier = SVC(C=0.5, kernel='linear', random_state=1234, verbose=verbose)

    classifier.fit(X_train, y_train)
    return classifier       

def test_classifier(X_test, y_test, classifier, n_labels=10):  
    n_examples = len(y_test)    
    confusion = np.zeros((n_labels,n_labels))

    for n, (X, true_label) in enumerate(zip(X_test,y_test)):
        sys.stdout.write('Classify progress: %2.0f%%\r' % (100*n/float(n_examples)))
        sys.stdout.flush()

        y_pred = np.array(classifier.predict(X), dtype='int')
        pred_label = np.argmax(np.bincount(y_pred, minlength=n_labels))
        confusion[pred_label, true_label] += 1
    print ''

    ave_acc = 100*(np.sum(np.diag(confusion)) / np.sum(confusion))
    print "classification accuracy:", ave_acc
    return confusion

def test_classifier_printf(X_test, y_test, Z_test, file_list, classifier, save_file, n_labels=10):
    n_examples = len(file_list)
    with open(save_file, 'w') as f:
        for n, (X, true_label, Z, fname) in enumerate(zip(X_test, y_test, Z_test, file_list)):
            sys.stdout.write('Classify progress: %2.0f%%\r' % (100*n/float(n_examples)))
            sys.stdout.flush()

            y_pred = np.array(classifier.predict(X), dtype='int')
            pred_label = np.argmax(np.bincount(y_pred, minlength=n_labels))
            s=''
            for i in Z: s+='%2.2f\t'%i
            f.write('{0}\t{1}\t{2}\t{3}\n'.format(fname, true_label, pred_label, s))
        print ''

if __name__ == "__main__":
    # example: python train_classifier_on_dnn_feats.py ./saved/S_500_RS.cpu.pkl /Users/cmke/Datasets/tzanetakis_genre --which_layers 0
    
    import argparse, glob

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description='''Script to train/test random forest on DNN features.
        ''')
    
    parser.add_argument('model_file', help='Path to trained DNN model file')
    parser.add_argument('dataset_dir', help='Path to dataset files (single directory with no subfolders)')
    parser.add_argument('--which_layers', nargs='*', type=int, help='List of which DNN layers to use as features')
    parser.add_argument('--aggregate_features', action='store_true', help='option to aggregate frames (mean/std of frames used to train classifier)')
    parser.add_argument('--save_folder', help='Output classification results to a text file')

    args = parser.parse_args()
    
    if not args.which_layers:
        parser.error('Please specify --which_layers x, with x either 0, 1, 2 or 0 1 2')

    if args.aggregate_features:
        print 'Using aggregate features'
    else:
        print 'Not using aggregate features'

    # load model
    model = serial.load(args.model_file) 

    # parse dataset
    p = re.compile(r"which_set.*'(train)'")
    trainset_yaml = model.dataset_yaml_src
    validset_yaml = p.sub("which_set: 'valid'", model.dataset_yaml_src)
    testset_yaml  = p.sub("which_set: 'test'", model.dataset_yaml_src)

    trainset = yaml_parse.load(trainset_yaml)
    validset = yaml_parse.load(validset_yaml)
    testset  = yaml_parse.load(testset_yaml)

    if args.aggregate_features:
        X_train, y_train, Z_train = aggregate_features(model, trainset, which_layers=args.which_layers)
    else:
        X_train, y_train, Z_train = get_features(model, trainset, which_layers=args.which_layers)

    # train data
    y_train = np.hstack([y*np.ones(len(X)) for X,y in zip(X_train, y_train)]) # upsample y (one label for each aggregated frame, instead of one label per song)
    X_train = np.vstack(X_train)
    
    # test data
    if args.aggregate_features:
        X_test, y_test, Z_test = aggregate_features(model, testset, which_layers=args.which_layers)
    else:
        X_test, y_test, Z_test = get_features(model, testset, which_layers=args.which_layers)
        
    print 'Training classifier'
    classifier = train_classifier(X_train, y_train, method='random_forest')    

    file_list=sorted(glob.glob(os.path.join(args.dataset_dir, '*.wav')))
    file_numbers = testset.raw.file_list

    print 'Testing classifier'
    if args.save_folder:    
        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)

        fname = os.path.split(args.save_folder)[-1]
        test_classifier_printf(X_test, y_test, Z_test, [os.path.split(file_list[i])[-1] for i in file_numbers], classifier, os.path.join(args.save_folder, fname+'.txt'))

        print 'Saving trained classifier'
        #with open(fname, 'w') as f:
        #    cPickle.dump(classifier, f)

        joblib.dump(classifier, os.path.join(args.save_folder, fname+'.pkl'), 9)

    else:
        confusion = test_classifier(X_test, y_test, classifier)

