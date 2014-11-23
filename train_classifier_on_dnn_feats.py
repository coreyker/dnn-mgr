import sys, re, cPickle
import numpy as np
import theano

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

    # fix---replace with song-level iterator from dataset....
    n_classes  = dataset.raw.y.shape[1]    
    n_examples = len(dataset.raw.support)
    n_frames_per_file   = dataset.raw.n_frames_per_file
    n_frames_per_sample = dataset.raw.n_frames_per_sample

    batch_size = n_frames_per_file // n_frames_per_sample
    #data_specs = dataset.raw.get_data_specs()
    feat_space   = model.get_input_space()
    target_space = VectorSpace(dim=n_classes)

    data_specs = (CompositeSpace((feat_space, target_space)), ("features", "targets")) 
    
    iterator   = dataset.iterator(mode='sequential', 
        batch_size=batch_size, #1, for song-level....
        data_specs=data_specs
        )
    # iterator = dataset.songlevel_iterator(mode='sequential', batch_size=1, data_specs=data_specs)

    # compute feature representation, aggregrate frames
    X=[]; y=[]; n=0
    for el in iterator:
        n += 1
        print n

        input_data  = np.array(el[0], dtype=np.float32)
        output_data = fprop(input_data)
        feats = np.hstack([output_data[i] for i in which_layers])

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

    return X, y

def train_classifier(X_train, y_train, method='random_forest'):
    assert method in ['random_forest', 'linear_svm']
    
    # train classifier
    if method=='random_forest':
        classifier = RandomForestClassifier(n_estimators=1000, random_state=1234)
    else:
        classifier = SVC(C=0.5, kernel='linear', random_state=1234)

    classifier.fit(X_train, y_train)
    return classifier       

def test_classifier(X_test, y_test, classifier):  
    confusion = np.zeros((10,10))
    
    for X, true_label in zip(X_test,y_test):
        y_pred = np.array(classifier.predict(X), dtype='int')
        pred_label = np.argmax(np.bincount(y_pred, minlength=10))
        confusion[pred_label, true_label] += 1

    ave_acc = 100*(np.sum(np.diag(confusion)) / np.sum(confusion))
    print "classification accuracy:", ave_acc
    return confusion

if __name__ == "__main__":

    # load model
    model_file = './saved/mlp_rlu-fold-1_of_4.cpu.pkl' #sys.argv[1]
    model = serial.load(model_file) 

    # parse dataset
    p = re.compile(r"which_set.*'(train)'")
    trainset_yaml = model.dataset_yaml_src
    validset_yaml = p.sub("which_set: 'valid'", model.dataset_yaml_src)
    testset_yaml  = p.sub("which_set: 'test'", model.dataset_yaml_src)

    trainset = yaml_parse.load(trainset_yaml)
    validset = yaml_parse.load(validset_yaml)
    testset  = yaml_parse.load(testset_yaml)

    which_layers = [2]
    X_train, y_train = aggregate_features(model, trainset, which_layers=which_layers)
    
    # train data
    y_train = np.hstack([y*np.ones(len(X)) for X,y in zip(X_train, y_train)]) # upsample y (one label for each aggregated frame, instead of one label per song)
    X_train = np.vstack(X_train)
    
    # test data
    X_test, y_test = aggregate_features(model, testset, which_layers=which_layers)
    
    # train then test:
    classifier = train_classifier(X_train, y_train, method='linear_svm') #'random_forest'):
    confusion = test_classifier(X_test, y_test, classifier)




