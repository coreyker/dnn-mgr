import os, sys, re, cPickle
import numpy as np
import theano

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from pylearn2.utils import serial
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import pylearn2.config.yaml_parse as yaml_parse

import pdb

def aggregate_features(model, dataset, which_layers=[2], win_size=200, step=100):
    assert np.max(which_layers) < len(model.layers)

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X, return_all=True)
    fprop = theano.function([X],Y)

    n_classes  = dataset.y.shape[1]
    n_examples = len(dataset.file_list)

    feat_space   = model.get_input_space()
    target_space = VectorSpace(dim=n_classes)

    data_specs = (CompositeSpace((feat_space, target_space)), ("songlevel-features", "targets"))     
    iterator = dataset.iterator(mode='sequential', data_specs=data_specs)

    # compute feature representation, aggregrate frames
    X=[]; y=[]; Z=[]; file_list=[];
    for n,el in enumerate(iterator):
        # display progress indicator
        sys.stdout.write('Aggregation progress: %2.0f%%\r' % (100*n/float(n_examples)))
        sys.stdout.flush()

        input_data  = np.array(el[0], dtype=np.float32)
        output_data = fprop(input_data)
        feats       = np.hstack([output_data[i] for i in which_layers])
        true_label  = el[1]

        # aggregate features
        agg_feat = []
        for i in xrange(0, feats.shape[0]-win_size, step):
            chunk = feats[i:i+win_size,:]
            agg_feat.append(np.hstack((np.mean(chunk, axis=0), np.std(chunk, axis=0))))
        
        X.append(np.vstack(agg_feat))
        y.append(np.hstack([true_label] * len(agg_feat)))
        Z.append(np.sum(output_data[-1], axis=0)) 
        file_list.append(el[2])

    print '' # newline
    return X, y, Z, file_list

def get_features(model, dataset, which_layers=[2], n_features=100):
    assert np.max(which_layers) < len(model.layers)

    rng = np.random.RandomState(111)
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X, return_all=True)
    fprop = theano.function([X],Y)

    n_classes  = dataset.y.shape[1]
    n_examples = len(dataset.file_list)

    feat_space   = model.get_input_space()
    target_space = VectorSpace(dim=n_classes)

    data_specs = (CompositeSpace((feat_space, target_space)), ("songlevel-features", "targets"))     
    iterator = dataset.iterator(mode='sequential', data_specs=data_specs)
    
    X=[]; y=[]; Z=[]; file_list=[];
    for n,el in enumerate(iterator):
        # display progress indicator
        sys.stdout.write('Getting features: %2.0f%%\r' % (100*n/float(n_examples)))
        sys.stdout.flush()

        input_data  = np.array(el[0], dtype=np.float32)
        output_data = fprop(input_data)        
        feats = np.hstack([output_data[i] for i in which_layers])
        true_label = el[1]

        if n_features:
            ind   = rng.permutation(feats.shape[0])
            feats = feats[ind[:n_features],:]

        X.append(feats)
        y.append([true_label]*n_features)
        Z.append(np.sum(output_data[-1], axis=0))
        file_list.append(el[2])

    print ''
    return X, y, Z, file_list

def train_classifier(X_train, y_train, method='random_forest', verbose=2):
    assert method in ['random_forest', 'linear_svm']
    
    # train classifier
    if method=='random_forest':
        classifier = RandomForestClassifier(n_estimators=500, random_state=1234, verbose=verbose, n_jobs=2)
    else:
        parameters = {'C' : 10**np.arange(-2,4.)}
        grid = GridSearchCV(SVC(), parameters, verbose=3)
        grid.fit(X_train, y_train)
        classifier = grid.best_estimator_
        #classifier = SVC(C=0.5, kernel='linear', random_state=1234, verbose=verbose)

    return classifier.fit(X_train, y_train)       

def test_classifier(X_test, y_test, classifier, n_labels=10):  
    n_examples = len(y_test)    
    confusion = np.zeros((n_labels,n_labels))

    for n, (X, true_label) in enumerate(zip(X_test,y_test)):
        sys.stdout.write('Classify progress: %2.0f%%\r' % (100*n/float(n_examples)))
        sys.stdout.flush()

        y_pred = np.array(classifier.predict(X), dtype='int')
        pred_label = np.argmax(np.bincount(y_pred, minlength=n_labels))
        confusion[pred_label, true_label[0]] += 1
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
            f.write('{0}\t{1}\t{2}\t{3}\n'.format(fname, true_label[0], pred_label, s))
        print ''

if __name__ == "__main__":
    # example: python train_classifier_on_dnn_feats.py ./saved/S_500_RS.cpu.pkl /Users/cmke/Datasets/tzanetakis_genre --which_layers 0
    
    import argparse, glob

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description='''Script to train/test random forest on DNN features.
        ''')
    
    parser.add_argument('model_file', help='Path to trained DNN model file')
    parser.add_argument('--which_layers', nargs='*', type=int, help='List of which DNN layers to use as features')
    parser.add_argument('--aggregate_features', action='store_true', help='option to aggregate frames (mean/std of frames used to train classifier)')
    parser.add_argument('--classifier', help="either 'random_forest' or 'linear_svm'")
    parser.add_argument('--save_file', help='Output classification results to a text file')

    args = parser.parse_args()
    
    if not args.which_layers:
        parser.error('Please specify --which_layers x, with x either 1, 2, 3 or 1 2 3 (layer 0 is a pre-processing layer)')

    if args.aggregate_features:
        print 'Using aggregate features'
    else:
        print 'Not using aggregate features'

    if args.classifier is None:
        print 'No classifer selected, using random forest'
        args.classifier = 'random_forest'

    # load model
    model = serial.load(args.model_file) 

    # parse dataset from model
    p = re.compile(r"which_set.*'(train)'")
    trainset_yaml = model.dataset_yaml_src
    validset_yaml = p.sub("which_set: 'valid'", model.dataset_yaml_src)
    testset_yaml  = p.sub("which_set: 'test'", model.dataset_yaml_src)

    trainset = yaml_parse.load(trainset_yaml)
    validset = yaml_parse.load(validset_yaml)
    testset  = yaml_parse.load(testset_yaml)

    if args.aggregate_features:
        X_train, y_train, Z_train, train_files = aggregate_features(model, trainset, which_layers=args.which_layers)
        X_valid, y_valid, Z_valid, valid_files = aggregate_features(model, validset, which_layers=args.which_layers)
        X_test, y_test, Z_test, test_files = aggregate_features(model, testset, which_layers=args.which_layers)
    else:
        X_train, y_train, Z_train, train_files = get_features(model, trainset, which_layers=args.which_layers)
        X_valid, y_valid, Z_valid, valid_files = get_features(model, validset, which_layers=args.which_layers)        
        X_test, y_test, Z_test, test_files = get_features(model, testset, which_layers=args.which_layers)
        
    print 'Training classifier'
    X_all = np.vstack((np.vstack(X_train), np.vstack(X_valid)))
    y_all = np.hstack((np.hstack(y_train), np.hstack(y_valid)))
    classifier = train_classifier(X_all, y_all, method=args.classifier)

    print 'Testing classifier'
    if args.save_file:    

        test_classifier_printf(
            X_test=X_test, 
            y_test=y_test, 
            Z_test=Z_test, 
            file_list=test_files, 
            classifier=classifier, 
            save_file=args.save_file+'.txt')

        print 'Saving trained classifier'
        joblib.dump(classifier, args.save_file+'.pkl', 9)

    else:
        confusion = test_classifier(X_test, y_test, classifier)

