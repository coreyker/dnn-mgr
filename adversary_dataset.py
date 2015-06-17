import numpy as np
import functools
import tables

from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrixPyTables, DefaultViewConverter
from pylearn2.blocks import Block
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
from pylearn2.utils.iteration import SubsetIterator, FiniteDatasetIterator, resolve_iterator_class
from pylearn2.utils import safe_zip, safe_izip

from pylearn2.datasets import control
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan
from pylearn2.models.mlp import MLP, Linear, PretrainedLayer
from pylearn2.models.autoencoder import Autoencoder

#from utils.test_adversary_linearized import find_adversary
import theano
from theano import config
from theano import tensor as T

import pdb

class AdversaryDataset(DenseDesignMatrixPyTables):
    def __init__(self, config, adv_model, which_set='train'):

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
        self.mean      = self.mean.reshape((np.prod(self.mean.shape),))
        self.var       = config['var']
        self.var       = self.var.reshape((np.prod(self.var.shape),))
        self.istd      = np.reciprocal(np.sqrt(self.var))
        self.mask      = (self.istd < 20)
        self.tframes   = config['tframes']

        # setup adversary
        self.adv_model = adv_model
        in_batch  = adv_model.get_input_space().make_theano_batch()
        out_batch = adv_model.get_output_space().make_theano_batch()        
        cost      = adv_model.cost(out_batch, adv_model.fprop(in_batch))
        dCost     = T.grad(cost, in_batch)

        grad_theano  = theano.function([in_batch, out_batch], dCost)
        fprop_theano = theano.function([in_batch], adv_model.fprop(in_batch))
        fcost_theano = theano.function([in_batch, out_batch], cost)

        self.input_space = adv_model.get_input_space()
        if isinstance(self.input_space, Conv2DSpace):
            tframes, dim   = self.input_space.shape
            view_converter = DefaultViewConverter((tframes, dim, 1))

            def grad(batch, labels):            
                 topo_view = grad_theano(view_converter.get_formatted_batch(batch, self.input_space), labels)
                 return view_converter.topo_view_to_design_mat(topo_view)

            def fprop(batch):
                 return fprop_theano(view_converter.get_formatted_batch(batch, self.input_space))

            def fcost(batch, labels):
                 return fcost_theano(view_converter.get_formatted_batch(batch, self.input_space), labels)

            self.grad = grad
            self.fprop = fprop
            self.fcost = fcost

            
            super(AdversaryDataset, self).__init__(X=data.X, y=data.y,
                view_converter=view_converter)

        else:
            dim = self.input_space.dim        
            tframes = 1
            view_converter = None

            self.grad = grad_theano
            self.fprop = fprop_theano
            self.fcost = fcost_theano 

            super(AdversaryDataset, self).__init__(X=data.X, y=data.y)
    
    def __del__(self):
        self.hdf5.close()   
    
    def create_adversary_from_batch(self, batch, label, mu=0.25, snr=15):
        n_examples = batch.shape[0]
        epsilon = np.linalg.norm(batch)/n_examples/10**(snr/20.)

        g = self.grad(batch, label) * n_examples
        Z = batch - mu * np.sign(g)
        Z = Z * (Z>0)

        nu = np.linalg.norm((Z-batch))/n_examples/epsilon - 1 # lagrange multiplier
        nu = nu * (nu>=0)
        Y  = (Z + nu*batch) / (1+nu)
        return Y
    
    def standardize(self, batch):
        return (batch - self.mean) * self.istd

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
                    X = np.abs(data[index:index+self._dataset.tframes, :])
                    design_mat.append( X.reshape((np.prod(X.shape),)) )
                   
                design_mat = np.vstack(design_mat)
                if self._dataset.tframes > 1:
                    # ideally we'd standardize in a preprocessing layer
                    # (so that standardization is built-in to the model rather
                    # than the dataset) but i haven't quite figured out how to do 
                    # this yet for images, due to a memory error associated with
                    # a really big diagonal scaling matrix
                    # (however, it works fine for vectors)
                    design_mat = self._dataset.standardize(design_mat)

                n_classes = len(self._dataset.targets)
                one_hot = np.zeros((design_mat.shape[0], n_classes), dtype=np.float32)
                for r in one_hot: r[np.random.randint(n_classes)] = 1
                
                design_mat = self._dataset.create_adversary_from_batch(design_mat, one_hot)
                
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
        
        thop = 1. # hardcoded and must match prepare_dataset.py!!!
        sup = np.arange(0,nframes-self._dataset.tframes,np.int(self._dataset.tframes/thop))        
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
                    else:
                        X = np.abs(data[index:index+self._dataset.tframes, :])
                    design_mat.append( X.reshape((np.prod(X.shape),)) )                    
                
                design_mat = np.vstack(design_mat)

                if self._dataset.tframes > 1:
                    # ideally we'd standardize in a preprocessing layer
                    # (so that standardization is built-in to the model rather
                    # than the dataset) but i haven't quite figured out how to do 
                    # this yet for images, due to a memory error associated with
                    # a really big diagonal scaling matrix
                    # (however, it works fine for vectors)
                    design_mat = self._dataset.standardize(design_mat)

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
    # should this use a linear layer instead of an autoencoder
    # (problem is layers don't implement upward_pass as required by pretrained layer...
    # but perhaps could write upward_pass function to call layer's fprop.)
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
            dim = nvis
            mask = (self.istd < 20) # in order to ignore near-zero variance inputs
            self.biases = np.array(-self.mean * self.istd * mask, dtype=np.float32)
            self.weights = np.array(np.diag(self.istd * mask), dtype=np.float32)
            
        if proc_type == 'pca_whiten':
            raise NotImplementedError(
            '''PCA whitening not yet implemented as a layer. 
            Use audio_dataset2d.AudioDataset2d to perform whitening from the dataset iterator''')

            # dim      = kwargs['ncomponents']
            # S        = config['S'][:dim]   # eigenvalues
            # U        = config['U'][:,:dim] # eigenvectors            
            # self.pca = np.diag(1./(np.sqrt(S) + epsilon)).dot(U.T)
            
            # self.biases   = np.array(-self.mean.dot(self.pca.transpose()), dtype=np.float32)
            # self.weights  = np.array(self.pca.transpose(), dtype=np.float32)

        # Autoencoder with linear units
        pre_layer = Autoencoder(nvis=nvis, nhid=dim, act_enc=None, act_dec=None, irange=0)
        
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


