import numpy as np
import functools
import tables

from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrixPyTables, DefaultViewConverter
from pylearn2.blocks import Block
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, VectorSequenceSpace, IndexSpace
from pylearn2.utils.iteration import SubsetIterator, FiniteDatasetIterator, resolve_iterator_class
from pylearn2.utils import safe_zip, safe_izip

from pylearn2.datasets import control
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan
from pylearn2.models.mlp import MLP, Linear, PretrainedLayer, CompositeLayer
from pylearn2.models.autoencoder import Autoencoder

from pylearn2.blocks import Block, StackedBlocks
from pylearn2.utils import as_floatX, safe_update, sharedX
from pylearn2.models import Model
from pylearn2.linear.matrixmul import MatrixMul

from theano import config
import theano
import theano.tensor as T

import pdb

class AudioDataset2d(DenseDesignMatrixPyTables):
    def __init__(self, config, which_set='train', ncomponents=80, epsilon=3):

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
                
        # PCA xform
        S        = config['S'][:ncomponents]   # eigenvalues
        U        = config['U'][:,:ncomponents] # eigenvectors            
        self.pca = np.diag(1./(np.sqrt(S) + epsilon)).dot(U.T)

        X = T.dmatrix('X')
        Z = T.transpose(T.dot(self.pca, T.transpose(X-self.mean)))
        self.transform = theano.function([X], Z)
        
        view_converter = DefaultViewConverter((self.tframes, ncomponents, 1))

        super(AudioDataset2d, self).__init__(X=data.X, 
            y=data.y,
            view_converter=view_converter)
    
    def __del__(self):
        self.hdf5.close()   
    
    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=1, num_batches=None,
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
            if (src == 'features' or src == 'songlevel-features') and \
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

        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng

        if 'songlevel-features' in sub_sources:
            if batch_size is not 1:
                raise ValueError("'batch_size' must be set to 1 for songlevel iterator")
            return SonglevelIterator2d(self,
                              mode(len(self.file_list), batch_size, num_batches, rng),
                              data_specs=data_specs,
                              return_tuple=return_tuple,
                              convert=convert)
        else:
            return FramelevelIterator2d(self,
                                  mode(len(self.support), batch_size, num_batches, rng),
                                  data_specs=data_specs,
                                  return_tuple=return_tuple,
                                  convert=convert)

class FramelevelIterator2d(FiniteDatasetIterator):
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
                    X = np.abs(data[index:index+self._dataset.tframes, :])
                    X = self._dataset.transform(X) # !!! PCA !!!

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

class SonglevelIterator2d(FiniteDatasetIterator):
    '''
    Returns all data associated with a particular song from the dataset
    (only iterates 1 song at a time!)
    '''
    @functools.wraps(SubsetIterator.next)
    def next(self):
        
        # next numerical index
        next_file_index = self._subset_iterator.next()        
        
        # associate numerical index with file from the dataset
        next_file = self._dataset.file_list[ next_file_index ][0] # !!! added line to iterate over different index set !!!
        
        # lookup file's position in the hdf5 array
        offset, nframes, key, target = self._dataset.file_index[next_file]

        thop = 10. # hardcoded and must match prepare_dataset.py!!!
        sup = np.arange(0,nframes-self._dataset.tframes,np.ceil(self._dataset.tframes/thop))
        next_index = offset + sup

        spaces, sources = self._data_specs
        output = []                

        for data, fn, source, space in safe_izip(self._raw_data, self._convert, sources, spaces.components):
            if source=='targets':
                # if fn:
                #     output.append( fn( np.reshape(data[next_index[0], :], (1,-1)) ) )
                # else:
                #     output.append( np.reshape(data[next_index[0], :], (1,-1)) )
                output.append( target )
            else:
                design_mat = []
                for index in next_index:
                    if space.dtype=='complex64':
                        X = data[index:index+self._dataset.tframes, :] # return phase too
                        X = self._dataset.transform(np.abs(X)) # !!! PCA ???!!!
                    else:
                        X = np.abs(data[index:index+self._dataset.tframes, :])
                        X = self._dataset.transform(X) # !!! PCA !!!

                    design_mat.append( X.reshape((np.prod(X.shape),)) )                    
                design_mat = np.vstack(design_mat)

                if fn:
                    output.append( fn(design_mat) )
                else:
                    output.append( design_mat )
        
        output.append(next_file)
        rval = tuple(output)
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval

class PreprocLayer(PretrainedLayer):
    def __init__(self, config, proc_type='standardize', **kwargs):
        '''
        config: dictionary with partition configuration information
        
        proc_type: type of preprocessing (either standardize or pca_whiten)
        
        if proc_type='standardize' no extra arguments required

        if proc_type='pca_whiten' the following keyword arguments are required:
            ncomponents = x where x is an integer
            epsilon = y where y is a float (regularization parameter)
        '''

        recognized_types = ['standardize', 'pca_whiten']
        assert proc_type in recognized_types

        # load parition information
        self.mean    = config['mean']
        self.istd    = np.reciprocal(np.sqrt(config['var']))
        self.tframes = config['tframes']
        nvis = len(self.mean)

        if proc_type == 'standardize':
            dim      = nvis
            self.biases   = np.array(-self.mean * self.istd, dtype=np.float32)
            self.weights  = np.array(np.diag(self.istd), dtype=np.float32)
        
            # Autoencoder with linear units
            pre_layer = Autoencoder(nvis=nvis, nhid=dim, act_enc=None, act_dec=None, irange=0)
            
        if proc_type == 'pca_whiten':
            raise NotImplementedError(
            '''PCA whitening not yet implemented as a layer. 
            Use audio_dataset2d.AudioDataset2d to perform whitening from the dataset iterator''')

            # - how to apply the same transform to each row of the input?
            # - setting nvis=nvis*tframes,  nhid=dim*tframes leads to out of memory error (and is wasteful because we don't need independent weights)
            # - seems a conv. autoencoder might work..., but this is not implemented in pylearn2...
            # - conv rbm may be possible, but this passes the output through a sigmoid (and we want a linear transform)
            # - could move the whitening to the dataset iterator (as before), but then the model can't be applied to new data without first somehow
            # accessing the whitening paramters, and preprocessing with them...
            # - A composite layer might work, but seems problematic right now...
            # - or use a ConvElemwise layer with weights set and then frozen... tried this in yaml file, but it is not cooperating...

            # Maybe the simplest thing to do would be to make an Autoencoder(Block, Model)-like class
            # that implements an upward_pass, and can be made to apply the transform to the individual rows? 
            # mlp.PreTrained layer just has to work, so whatever is minimally required there should be fine?

            # in_space  = Conv2DSpace(shape=(self.tframes, nvis), num_channels=1, axes=('b', 0, 1, 'c'))
            # out_space = Conv2DSpace(shape=(self.tframes, dim),  num_channels=1, axes=('b', 0, 1, 'c'))
            # pre_layer = PCAWhitener(in_space, out_space, self.weights)

        # Set weights for pre-processing
        params    = pre_layer.get_param_values()
        params[1] = self.biases
        params[2] = self.weights
        pre_layer.set_param_values(params)

        super(PreprocLayer, self).__init__(layer_name='pre', layer_content=pre_layer, freeze_params=True)        
    
    def get_biases(self):
        return self.biases

    def get_weights(self):
        return self.weights

    def get_param_values(self):
        return list((self.get_weights(), self.get_biases()))

# class PCAWhitener(Block, Model):
#     def __init__(self, vis_space, hid_space, weights):#, W, b):
#         Model.__init__(self)
#         Block.__init__(self)

#         self.vis_space = vis_space
#         self.hid_space = hid_space
#         #self.b = b
#         self.transformer = MatrixMul(  sharedX(
#                 weights,
#                 name='W',
#                 borrow=True
#             )
#         )

#     def get_input_space(self):
#         return self.vis_space

#     def get_output_space(self):
#         return self.hid_space

#     def upward_pass(self, x):
#         #pdb.set_trace()
#         if isinstance(x, tensor.Variable):
#            return self.transformer.lmul_T(x)
#         else:
#            return [self.upward_pass(i) for i in x]        

# class PCALayer(CompositeLayer):
#     def __init__(self, config, n_components):
        
#         # load parition information
#         self.mean    = config['mean']
#         self.istd    = np.reciprocal(np.sqrt(config['var']))
#         self.tframes = config['tframes']
#         nvis = len(self.mean)

#         layer = Linear(dim=n_components, layer_name='pca', irange=0)
#         return super(PCALayer, self).__init__(layer_name='pre', layers=[layer for t in xrange(self.tframes)])


if __name__=='__main__':

    # tests
    import theano
    import cPickle
    from audio_dataset import AudioDataset2d

    with open('GTZAN_stratified.pkl') as f: 
        config = cPickle.load(f)
    
    D = AudioDataset2d(config)
    
    feat_space   = VectorSpace(dim=D.X.shape[1])
    feat_space_complex = VectorSpace(dim=D.X.shape[1], dtype='complex64')
    target_space = VectorSpace(dim=len(D.label_list))
    
    data_specs_frame = (CompositeSpace((feat_space,target_space)), ("features", "targets"))
    data_specs_song = (CompositeSpace((feat_space_complex, target_space)), ("songlevel-features", "targets"))

    framelevel_it = D.iterator(mode='sequential', batch_size=10, data_specs=data_specs_frame)
    frame_batch = framelevel_it.next()

    songlevel_it = D.iterator(mode='sequential', batch_size=1, data_specs=data_specs_song)    
    song_batch = songlevel_it.next()

