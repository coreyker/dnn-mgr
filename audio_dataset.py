import numpy as np
import functools
import tables

from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrixPyTables
from pylearn2.blocks import Block
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
from pylearn2.utils.iteration import SubsetIterator, FiniteDatasetIterator, resolve_iterator_class
from pylearn2.utils import safe_zip, safe_izip

from pylearn2.datasets import control
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan
from pylearn2.models.mlp import MLP, Linear

from theano import config

import pdb

class AudioDataset(DenseDesignMatrixPyTables):
    def __init__(self, config, which_set='train', standardize=True, pca_whitening=False, ncomponents=None, epsilon=3):

        keys = ['train', 'test', 'valid']
        assert which_set in keys

        # load hdf5 metadata
        self.hdf5       = tables.open_file( config['hdf5'], mode='r')
        data            = self.hdf5.get_node('/', 'Data')
        param           = self.hdf5.get_node('/', 'Param')
        self.file_index = param.file_index[0]
        self.file_dict  = param.file_dict[0]
        self.label_list = param.label_list[0]
        self.targets    = param.targets[0]
        self.nfft       = param.fft[0]['nfft']

        # load parition information
        self.support   = config[which_set]
        self.file_list = config[which_set+'_files']
        self.mean      = config['mean']
        self.var       = config['var']
        self.tframes   = config['tframes']

        # if (standardize is True) and (pca_whitening is True):
        #     raise ValueError("'standardize' and 'pca_whiten' cannot both be True")
        
        # if ncomponents is None:
        #     self.ncomponents = len(self.mean)
        # else:
        #     assert ncomponents <= len(self.mean)
        #     self.ncomponents = ncomponents

        # if (pca_whitening is True):
        #     S = config['S'][:self.ncomponents]   # eigenvalues
        #     U = config['U'][:,:self.ncomponents] # eigenvectors            
        #     self.pca = np.diag(1./(np.sqrt(S) + epsilon)).dot(U.T) 

        # # create linear layer to take care of pre-processing (e.g., standardization or whitening)
        # pre_layer = Linear(dim=self.ncomponents, layer_name='pre', irange=0, W_lr_scale=0, b_lr_scale=0)
        # m = MLP(nvis=self.nfft//2+1, layers=[preproc_layer]) # define input layer

        # if standardize is True:
        #     pre_layer.set_biases(np.array(-self.mean/self.var, dtype=np.float32))
        #     pre_layer.set_weights(np.diag(np.reciprocal(self.var), dtype=np.float32))
        #     # self.transform = lambda X: (X-self.mean)/self.var
        # elif pca_whitening is True:
        #     pre_layer.set_biases(np.array(-self.mean.dot(self.pca.transpose()), dtype=np.float32))
        #     pre_layer.set_weights(np.array(self.pca.transpose(), dtype=np.float32))
        #     #self.transform = lambda X: (X-self.mean).dot(self.pca.transpose())
        # else:
        #     pass
        #     #self.transform = lambda X: X

        super(AudioDataset, self).__init__(X=data.X, y=data.y)
    
    def __del__(self):
        self.hdf5.close()   
    
    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):
        '''
        Copied from pylearn2 superclass in order to return custom iterator.
        Two different iterators are available, depending on the data_specs.
        1. If the data_specs source is 'features' a framelevel iterator is returned 
        (each call to next() returns a single frame)
        2. If the data_specs source is 'songlevel-features' a songlevel iterator is returned
        (each call to next() returns all the frames associated with a given song in the dataset)
        '''
        if data_specs is None:
            data_specs = self._iter_data_specs
        else:
            self.data_specs = data_specs

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            if src == 'features' and \
               getattr(self, 'view_converter', None) is not None:
                conv_fn = (lambda batch, self=self, space=sp:
                           self.view_converter.get_formatted_batch(batch,
                                                                   space))
            else:
                conv_fn = None

            convert.append(conv_fn)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng

        if 'songlevel-features' in sub_sources:
            return SonglevelIterator(self,
                              mode(len(self.file_list), batch_size, num_batches, rng),
                              data_specs=data_specs,
                              return_tuple=return_tuple,
                              convert=convert)
        else:
            return FramelevelIterator(self,
                                  mode(len(self.support), batch_size, num_batches, rng),
                                  data_specs=data_specs,
                                  return_tuple=return_tuple,
                                  convert=convert)

class FramelevelIterator(FiniteDatasetIterator):
    '''
    Returns individual (spectrogram) frames/slices from the dataset
    '''
    @functools.wraps(SubsetIterator.next)
    def next(self):
        """
        Retrieves the next batch of examples.

        Returns
        -------
        next_batch : object
            An object representing a mini-batch of data, conforming
            to the space specified in the `data_specs` constructor
            argument to this iterator. Will be a tuple if more
            than one data source was specified or if the constructor
            parameter `return_tuple` was `True`.

        Raises
        ------
        StopIteration
            When there are no more batches to return.
        """
        next_index = self._subset_iterator.next()
        next_index = self._dataset.support[ next_index ] # !!! added line to iterate over different index set !!!

        spaces, sources = self._data_specs
        output = []                

        for data, fn, source in safe_izip(self._raw_data, self._convert, sources):
            if source=='targets':
                if fn:
                    output.append( fn(data[next_index, :]) )
                else:
                    output.append( data[next_index, :] )
            else:
                design_mat = []
                for index in next_index:
                    #X = self._dataset.transform( data[index:index+self._dataset.tframes, :] )
                    X = data[index:index+self._dataset.tframes, :]
                    design_mat.append( X.reshape((np.prod(X.shape),)) )                    
                design_mat = np.vstack(design_mat)

                if fn:
                    output.append( fn(design_mat) )
                else:
                    output.append( design_mat )

        rval = tuple(output)
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval

class SonglevelIterator(FiniteDatasetIterator):
    '''
    Returns all data associated with a particular song from the dataset
    '''
    @functools.wraps(SubsetIterator.next)
    def next(self):
        
        # next numerical index
        next_file_index = self._subset_iterator.next()        
        
        # associate numerical index with file from the dataset
        next_file = self._dataset.file_list[ next_file_index ] # !!! added line to iterate over different index set !!!
        
        # lookup file's position in the hdf5 array
        next_index = []
        for f in next_file:
            offset, nframes, key, target = self._dataset.file_index[f]
            next_index.append(offset + np.arange(nframes))
        next_index = np.hstack(next_index)


        spaces, sources = self._data_specs
        output = []                

        for data, fn, source in safe_izip(self._raw_data, self._convert, sources):
            if source=='targets':
                if fn:
                    output.append( fn(data[next_index, :]) )
                else:
                    output.append( data[next_index, :] )
            else:
                design_mat = []
                for index in next_index:
                    #X = self._dataset.transform( data[index:index+self._dataset.tframes, :] )
                    X = data[index:index+self._dataset.tframes, :]
                    design_mat.append( X.reshape((np.prod(X.shape),)) )                    
                design_mat = np.vstack(design_mat)

                if fn:
                    output.append( fn(design_mat) )
                else:
                    output.append( design_mat )
                    
        rval = tuple(output)
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval

class PreprocLayer(PretrainedLayer):
    def __init__(self, config, proc_type='standardize', **kwargs):
        '''
        config: dictionary with partition configuration information
        
        proc_type: type of preprocessing (either standardize or pca_whiten)
        
        **kwargs list of key words and their arguments
        
        if proc_type='standardize' no extra arguments required

        if proc_type='pca_whiten' the following arguments are required:
            ncomponents = x where x is an integer
            epsilon = y where y is a float (regularization parameter)
        '''

        recognized_types = ['standardize', 'pca_whiten']
        assert proc_type in recognized_types

        # load parition information
        self.mean    = config['mean']
        self.var     = config['var']
        self.tframes = config['tframes']
        nvis = len(self.mean)

        if proc_type is 'standardize':
            dim = nvis
            biases  = np.array(-self.mean/self.var, dtype=np.float32)
            weights = np.diag(np.reciprocal(self.var), dtype=np.float32)
        
        if proc_type is 'pca_whiten':
            dim = kwargs['ncomponents']
            S = config['S'][:dim]   # eigenvalues
            U = config['U'][:,:dim] # eigenvectors            
            self.pca = np.diag(1./(np.sqrt(S) + epsilon)).dot(U.T)

            biases  = np.array(-self.mean.dot(self.pca.transpose()), dtype=np.float32)
            weights = np.array(self.pca.transpose(), dtype=np.float32)

        # create linear layer to take care of pre-processing
        pre_layer = Linear(dim=self.ncomponents, layer_name='pre', irange=0, W_lr_scale=0, b_lr_scale=0)
        pre_model = MLP(nvis=nvis, layers=[preproc_layer]) # define input layer

        pre_layer.set_biases(biases)
        pre_layer.set_weights(weights)

        super(PreprocLayer, self).__init__(layer_name='pre', layer_content=pre_model, freeze_params=True)        

if __name__=='__main__':

    # tests
    import cPickle

    with open('/Users/cmke/Datasets/tzanetakis_genre/tzanetakis_genre-fold-1_of_4.pkl') as f: 
        config = cPickle.load(f)
    
    D = AudioDataset(config, standardize=False, pca_whitening=True, ncomponents=50)
    
    feat_space   = VectorSpace(dim=D.ncomponents)    
    target_space = VectorSpace(dim=len(D.label_list))
    
    data_specs_frame = (CompositeSpace((feat_space,target_space)), ("features", "targets"))
    data_specs_song = (CompositeSpace((feat_space,target_space)), ("songlevel-features", "targets"))

    framelevel_it = D.iterator(mode='sequential', batch_size=10, data_specs=data_specs_frame)
    frame_batch = framelevel_it.next()

    songlevel_it = D.iterator(mode='sequential', batch_size=1, data_specs=data_specs_song)    
    song_batch = songlevel_it.next()

