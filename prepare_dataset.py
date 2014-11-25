"""
Prepare dataset:
	Compute fft, write to h5 file
"""
import os, sys, glob, tables
import numpy as np
import theano
import cPickle
from scikits import audiolab

import pdb

def make_h5_dataset(dataset_dir, h5_file_name='GTZAN_1024.h5', n_fft=1024, n_hop=512, n_frames_per_file=1200):

	if os.path.exists( h5_file_name ):	
		print h5_file_name + ' already exists, aborting...'
		return

	classes = {'blues':0, 'classical':1, 'country':2, 'disco':3, 'hiphop':4, 'jazz':5, 'metal':6, 'pop':7, 'reggae':8, 'rock':9}

	files_list = sorted(glob.glob( os.path.join( dataset_dir, '*.wav') ))

	h5file  = tables.open_file(h5_file_name, mode = "w", title = "GTZAN")
	filters = tables.Filters(complib='blosc', complevel=9)
	node    = h5file.create_group(h5file.root, "Data", "Data")
	atom    = tables.Float32Atom() if theano.config.floatX == 'float32' else tables.Float64Atom()

	h5file.create_earray(node, 'X', atom = atom, shape = (0,n_fft/2+1),
	                            title = "Data values", filters = filters)
	h5file.create_earray(node, 'y', atom = atom, shape = (0,10),
	                            title = "Data targets", filters = filters)

	window = np.hanning(n_fft)

	for n, file_handle in enumerate(files_list):

		print 'Computing FFT on file %d of %d' % (n+1, len(files_list))		

		cat = classes[ os.path.split(file_handle)[-1].split('.')[0] ] # get category from filename
		one_hot  = np.zeros( (n_frames_per_file, 10) )
		one_hot[:,cat] = 1

		audio_data, _, _  = audiolab.wavread(file_handle)
		fft_data = np.zeros( (n_frames_per_file, n_fft) )

		for k in xrange(n_frames_per_file):
			ptr = k * n_hop
			fft_data[k,:]  = audio_data[ptr:ptr+n_fft] * window          
		fft_data = np.abs( np.fft.fft(fft_data) )

		node.X.append( fft_data[:,:n_fft/2+1] )
		node.y.append( one_hot )

		h5file.flush()

	h5file.close()
	print '' # newline

def generate_fold_configs(h5_file_name='GTZAN_1024.h5', n_folds=4, valid_prop=0.333, n_frames_per_sample=1):
	"""	
	Generate dataset config files with stratified folds
	"""	
	
	n_files = 1000
	n_classes = 10

	# overall size of partition
	n_test_files  = n_files//n_folds
	n_valid_files = int( (n_files-n_test_files) * valid_prop + 0.5 )
	n_train_files = n_files - n_test_files - n_valid_files

	# get number of files per class (for stratified partition)
	n_test_files  = n_test_files // n_classes
	n_valid_files = n_valid_files // n_classes
	n_train_files = n_train_files // n_classes
	n_files_per_class = n_files // n_classes

	# load h5file
	h5file  = tables.open_file( h5_file_name, mode = "r" )
	data    = h5file.get_node('/', "Data")
	n_feats = data.X.shape[1]
	n_frames_per_file = data.X.shape[0]//n_files

	rng  = np.random.RandomState(314)
	perm = [rng.permutation(n_files_per_class) + n*n_files_per_class for n in xrange(n_classes)]


	for fold in xrange(n_folds):

		print 'Creating fold %d of %d' % (fold+1, n_folds)
		if n_frames_per_sample==1:
			config_file = os.path.splitext(h5_file_name)[0] + '-fold-%d_of_%d.pkl' % (fold+1, n_folds)
		else:
			config_file = os.path.splitext(h5_file_name)[0] + '-%d-fold-%d_of_%d.pkl' % (n_frames_per_sample, fold+1, n_folds)
		
		if os.path.exists( config_file ):	
			print config_file + ' already exists, aborting...'
			continue

		test_files    = n_classes*[None]
		train_files   = n_classes*[None]
		valid_files   = n_classes*[None]
		test_support  = n_classes*[None]
		train_support = n_classes*[None]
		valid_support = n_classes*[None]
		
		# get equal number of train/test/valid samples from each class (stratified)
		for n in xrange(n_classes):

			rot_perm = rotate(perm[n], fold * n_test_files)

			test_files[n]  = sorted(rot_perm[: n_test_files])
			train_files[n] = sorted(rot_perm[n_test_files : n_test_files + n_train_files])
			valid_files[n] = sorted(rot_perm[n_test_files + n_train_files:])

			test_support[n]  = np.hstack([i * n_frames_per_file + np.arange(0,n_frames_per_file,n_frames_per_sample) for i in test_files[n]])
			train_support[n] = np.hstack([i * n_frames_per_file + np.arange(0,n_frames_per_file,n_frames_per_sample) for i in train_files[n]])
			valid_support[n] = np.hstack([i * n_frames_per_file + np.arange(0,n_frames_per_file,n_frames_per_sample) for i in valid_files[n]])
		
		test_files    = sum(test_files,[])
		train_files   = sum(train_files,[])
		valid_files   = sum(valid_files,[])
		test_support  = np.hstack(test_support)
		train_support = np.hstack(train_support)
		valid_support = np.hstack(valid_support)

		# compute mean and std for training set only
		#sum_x  = np.zeros((n_frames_per_sample,n_feats), dtype=np.float32)
		#sum_x2 = np.zeros((n_frames_per_sample,n_feats), dtype=np.float32)	
		sum_x  = np.zeros(n_feats, dtype=np.float32)
		sum_x2 = np.zeros(n_feats, dtype=np.float32)	
		n_samples = len(train_support)
		
		for n,i in enumerate(train_support):
			for j in xrange(n_frames_per_sample):
				sys.stdout.write('\rProgress: %2.2f%%' % (n/float(n_samples)*100))
				sys.stdout.flush()
				
			#fft_frame = data.X[i:i+n_frames_per_sample,:]
				fft_frame = data.X[i+j,:]
				sum_x  += fft_frame
				sum_x2 += fft_frame**2
		print ''

		mean = sum_x / n_samples
		var  = (sum_x2 - sum_x**2/n_samples)/(n_samples-1)

		# compute PCA whitening matrix
		XX = 0
		for n,i in enumerate(train_support):
			for j in xrange(n_frames_per_sample):
				sys.stdout.write('\rProgress: %2.2f%%' % (n/float(n_samples)*100))
				sys.stdout.flush()
				
			#fft_frame = data.X[i:i+n_frames_per_sample,:]
				fft_frame = data.X[i+j,:]
				X = np.reshape(fft_frame - mean, (len(fft_frame), 1))# / np.sqrt(var)
				XX += X.dot(X.T)
		print ''
		XX /= len(train_support)

		U,S,V = np.linalg.svd(XX)
		#PCA_xform = (1./(np.sqrt(S) + epsilon)).dot(U.T)

		config = {
			'h5_file_name': h5_file_name,
			'n_frames_per_file': n_frames_per_file,
			'n_frames_per_sample' : n_frames_per_sample,
			'test': test_support, 
			'train': train_support, 
			'valid': valid_support,
			'test_files': test_files,
			'train_files': train_files,
			'valid_files': valid_files,
			'mean': mean,
			'std': np.sqrt(var),
			'U' : U, # rot matrix
			'S' : S # eigenvalues
			}

		# pickle config		
		with open(config_file, 'w') as f:
			cPickle.dump( config, f, protocol=2 )

	h5file.close()

def generate_folds_from_files(h5_file_name=None, train_file=None, valid_file=None, test_file=None, n_frames_per_sample=1):

	assert h5_file_name is not None
	assert train_file is not None
	assert valid_file is not None
	assert test_file is not None

	if n_frames_per_sample==1:
		config_file = os.path.splitext(h5_file_name)[0] + '-filtered-fold.pkl'
	else:
		config_file = os.path.splitext(h5_file_name)[0] + '-%d-filtered-fold.pkl' % n_frames_per_sample

	if os.path.exists( config_file ):	
		print config_file + ' already exists, aborting...'
		return

	n_files = 1000
	n_classes = 10

	h5file  = tables.open_file( h5_file_name, mode = "r" )
	data    = h5file.get_node('/', "Data")
	n_feats = data.X.shape[1]
	n_frames_per_file = data.X.shape[0]//n_files

	with open(train_file) as f:
		lines = f.readlines()
		train_files = [int(i)-1 for i in lines]

	with open(valid_file) as f:
		lines = f.readlines()
		valid_files = [int(i)-1 for i in lines]

	with open(test_file) as f:
		lines = f.readlines()
		test_files = [int(i)-1 for i in lines]				

	test_support  = np.hstack([i * n_frames_per_file + np.arange(0,n_frames_per_file,n_frames_per_sample) for i in test_files])
	train_support = np.hstack([i * n_frames_per_file + np.arange(0,n_frames_per_file,n_frames_per_sample) for i in train_files])
	valid_support = np.hstack([i * n_frames_per_file + np.arange(0,n_frames_per_file,n_frames_per_sample) for i in valid_files])

	# compute mean and std for training set only
	sum_x  = np.zeros((n_frames_per_sample,n_feats), dtype=np.float32)
	sum_x2 = np.zeros((n_frames_per_sample,n_feats), dtype=np.float32)
	n_samples = len(train_support)
	
	for n,i in enumerate(train_support):
		sys.stdout.write('\rProgress: %2.2f%%' % (n/float(n_samples)*100))
		sys.stdout.flush()
		
		fft_frame = data.X[i:i+n_frames_per_sample,:]
		sum_x  += fft_frame
		sum_x2 += fft_frame**2
	print ''

	mean = sum_x / n_samples
	var  = (sum_x2 - sum_x**2/n_samples)/(n_samples-1)

	config = {
		'h5_file_name': h5_file_name,
		'n_frames_per_file': n_frames_per_file,
		'n_frames_per_sample' : n_frames_per_sample,
		'test': test_support, 
		'train': train_support, 
		'valid': valid_support,
		'test_files': test_files,
		'train_files': train_files,
		'valid_files': valid_files,
		'mean': mean,
		'std': np.sqrt(var)
		}

	# pickle config		
	with open(config_file, 'w') as f:
		cPickle.dump( config, f, protocol=2 )

def rotate(x, n):
	assert n<len(x)
	return np.hstack([x[n:], x[:n]])

if __name__ == "__main__":
	
	dataset_dir = sys.argv[1]	
	h5_file_name = 'GTZAN_1024.h5'
	
	make_h5_dataset(dataset_dir, h5_file_name)

	generate_fold_configs(h5_file_name)

	generate_folds_from_files(h5_file_name, 'train_filtered.txt', 'valid_filtered.txt', 'test_filtered.txt')
	
	#generate_folds_from_files(h5_file_name, 'train_filtered.txt', 'valid_filtered.txt', 'valid_filtered.txt') #'test_filtered_1perartist.txt')

	generate_fold_configs(h5_file_name, n_frames_per_sample=40)

	#generate_folds_from_files(h5_file_name, 'train_filtered.txt', 'valid_filtered.txt', 'test_filtered.txt', n_frames_per_sample=40)

