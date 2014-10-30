# svm train / test
import os, sys, copy, cPickle
import numpy as np
import theano
import theano.tensor as T
from pylearn2.utils import serial
from sklearn.svm import LinearSVC, SVC
import pdb

def train_svm(X,y,C):
	if 0:
		svm = LinearSVC(C=C, loss='l1', random_state=1234)
	else:
		svm = SVC(C=C, kernel='linear', random_state=1234)
	return svm.fit(X,y)
	
def test_svm(X, y, svm):
	n_classes = 10

	confusion = np.zeros((n_classes, n_classes))
	for feats, label in zip(X,y):
		true_label = label if np.isscalar(label) else label[0]
		pred       = np.array( svm.predict(feats), dtype='int' )
		vote_label = np.argmax( np.bincount(pred, minlength=10) )
		
		confusion[true_label, vote_label] += 1

	total_error = 100*(1 - np.sum(np.diag(confusion)) / np.sum(confusion))

	return total_error, confusion

def grid_search(X_train, y_train, X_valid, y_valid, C_values):
	n_classes  = 10
	best_svm   = None
	best_C     = None
	best_error = 100.
	best_conf  = None

	for m,C in enumerate(C_values):

		svm = train_svm( np.vstack(X_train), np.hstack(y_train), C)
		err, conf = test_svm( np.vstack(X_valid), np.hstack(y_valid), svm )
		#err, conf = test_svm(X_valid, y_valid, svm)

		if err < best_error:
			best_error = err
			best_svm   = copy.deepcopy(svm)
			best_C     = C
			best_conf  = conf

		print 'Model selection progress %2d%%, best_error=%2.2f, curr_error=%2.2f' % ((100*m)/len(C_values), best_error, err)

	print ''#newline
	return best_error, best_svm, best_C, best_conf

if __name__ == "__main__":

	# load in BOF features
	model = 'mlp_rlu_fold1_best'
	train_BOF = model + '-train-BOF.pkl'
	valid_BOF = model + '-valid-BOF.pkl'
	test_BOF  = model + '-test-BOF.pkl'

	with open(train_BOF) as f:
		X_train, y_train = cPickle.load(f)

	with open(valid_BOF) as f:
		X_valid, y_valid = cPickle.load(f)

	with open(test_BOF) as f:
		X_test, y_test = cPickle.load(f)

	C_values = 10.0 ** np.arange(-3, 3, 0.25)
	#C_values = np.arange(0.5, 1, 0.01)
	best_error, best_svm, best_C, best_conf = grid_search(X_train, y_train, X_valid, y_valid, C_values)

	# re-train on train+valid sets before testing
	best_svm = train_svm(np.vstack(sum([X_train, X_valid],[])), np.hstack(sum([y_train, y_valid],[])), best_C)
	total_error, confusion = test_svm(X_test, y_test, best_svm)

	print 'test accuracy: %2.2f' % (100-total_error)
	print 'confusion matrix:'
	print confusion/np.sum(confusion, axis=1)



