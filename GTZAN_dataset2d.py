import numpy as np
import functools
import tables
import theano
import theano.tensor as T
from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrixPyTables, DefaultViewConverter
from pylearn2.blocks import Block
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
from pylearn2.utils.iteration import SubsetIterator, FiniteDatasetIterator, resolve_iterator_class
from pylearn2.utils import safe_zip, safe_izip

from pylearn2.datasets import control
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan
from theano import config

import pdb

class GTZAN_iterator2d(FiniteDatasetIterator):

    def __init__(self):

        # compile theano function for pca whitening transform
        X = T.dmatrix('X')
        Z = T.transpose(T.dot(self._dataset.pca_xform, T.transpose(X-self._dataset.mean)))
        self.transform = theano.function([X], Z)

        super(GTZAN_iterator2d, self).__init__()

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

        # !!! fancy indexing doesn't seem to work w/ pytables (data[next_index] doesn't work, but data[next_index,:] does) !!!
        # rval = tuple(
        #     fn(data[next_index,:]) if fn else data[next_index,:] 
        #     for data, fn in safe_izip(self._raw_data, self._convert))
        # if not self._return_tuple and len(rval) == 1:
        #     rval, = rval
        # return rval

        # alternative for 2d data topology
        n_frames_per_sample = self._dataset.n_frames_per_sample
        spaces, sources = self._data_specs
        output = []                

        for data, fn, source in safe_izip(self._raw_data, self._convert, sources):
            if source=='targets':
                if fn:
                    output.append( fn(data[next_index, :]) )
                else:
                    output.append( data[next_index, :] )

            elif source=='features':
                design_mat = []
                for index in next_index:
                    sample  = data[index:index+n_frames_per_sample,:]

                    # do pca whitening here (instead of in transformer dataset)
                    #pca_sample = self._dataset.pca_xform.dot((sample-self._dataset.mean).transpose())
                    #pca_sample = pca_sample.transpose()                    
                    pca_sample = self.transform(sample)
                    
                    design_mat.append( pca_sample.reshape((np.prod(pca_sample.shape),)) )
                    

                    #design_mat.append( sample.reshape((np.prod(sample.shape),)) )
                    
                design_mat = np.vstack(design_mat)

                if fn:
                    output.append( fn(design_mat) )
                else:
                    output.append( design_mat )

            else:
                raise ValueError('Encountered unrecognized data source: %s' % source)

        rval = tuple(output)
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval

class GTZAN_dataset2d(DenseDesignMatrixPyTables):

    def __init__(self, config, which_set='train', n_components=80, epsilon=3):

        keys = ['train', 'test', 'valid']
        assert which_set in keys

        # load h5file
        self.h5file  = tables.open_file( config['h5_file_name'], mode = "r" )
        data         = self.h5file.get_node('/', "Data")

        self.support = config[which_set]
        self.n_frames_per_file   = config['n_frames_per_file']
        self.n_frames_per_sample = config['n_frames_per_sample']
        
        self.n_components = n_components        
        self.epsilon = epsilon

        # ...this doesn't work:
        self.mean = np.array(config['mean'], dtype=np.float32)
        self.std  = np.array(config['std'], dtype=np.float32)

        # PCA_xform
        S = config['S'][:n_components] # eigenvalues
        U = config['U'][:,:n_components]
        self.pca_xform = np.diag(1./(np.sqrt(S) + epsilon)).dot(U.T) 

        #!!!nb: 513 shouldn't be hardcoded here!!!
        view_converter = DefaultViewConverter((self.n_frames_per_sample, n_components, 1))
        
        super(GTZAN_dataset2d, self).__init__(X=data.X, y=data.y,
            view_converter=view_converter)

    def __del__(self):
        self.h5file.close()        

    # !!! copied from superclasqs so that we can return a GTZAN_iterator instead of FiniteDatasetIterator !!!
    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        if data_specs is None:
            data_specs = self._iter_data_specs

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

        return GTZAN_iterator2d(self,
                              mode(len(self.support), batch_size, num_batches, rng),
                              data_specs=data_specs,
                              return_tuple=return_tuple,
                              convert=convert)

# class GTZAN_standardizer2d(Block):

#     def __init__(self, config, n_components=80, epsilon=3):

#         self.n_components = n_components        
#         self.epsilon = epsilon

#         # ...this doesn't work:
#         self._mean = np.array(config['mean'], dtype=np.float32)
#         self._std  = np.array(config['std'], dtype=np.float32)
#         #self.input_space = Conv2DSpace(shape=self._mean.shape, num_channels=1, axes=('b', 'c', 0, 1))

#         # shape = (np.prod(config['mean'].shape), )
#         # self._mean = np.reshape(np.array(config['mean'], dtype=np.float32), shape)
#         # self._std  = np.reshape(np.array(config['std'], dtype=np.float32), shape)
#         # self.input_space = VectorSpace(dim=shape[0])
        
#         self.n_frames_per_sample = config['n_frames_per_sample']
#         self.input_space = VectorSpace(dim=self.n_frames_per_sample*len(self._mean))
#         #self.input_space = Conv2DSpace(shape=(self.n_frames_per_sample, len(self._mean)), num_channels=1, axes=('b', 'c', 0, 1))
#         #self.input_space = Conv2DSpace(shape=(self.n_frames_per_sample, len(self._mean)), num_channels=1, axes=('b', 'c', 0, 1))
        
#         # PCA_xform
#         S = config['S'][:n_components] # eigenvalues
#         U = config['U'][:,:n_components]
#         self.PCA_xform = np.diag(1./(np.sqrt(S) + epsilon)).dot(U.T)        

#         super(GTZAN_standardizer2d, self).__init__()

#     def __call__(self, batch):
#         """
#         .. todo::

#             WRITEME
#         """
#         if self.input_space:
#             self.input_space.validate(batch)

        
#         design_mat = []#np.zeros(len(batch), self.n_frames_per_sample*self.n_components)
#         for i,ex in enumerate(batch):

#             # 1. convert to matrix view of spectrogram
#             ex = np.reshape(ex, (self.n_frames_per_sample, len(self._mean)))
#             # 2. apply pca
#             ex = self.PCA_xform.dot(ex.T)
#             # 3. reshape back to column vec
#             #design_mat[i] = np.reshape(ex.T, (self.n_frames_per_sample*self.n_components,))
#             design_mat.append( np.reshape(ex.T, (self.n_frames_per_sample*self.n_components,)) )
#         return design_mat

#     def set_input_space(self, space):
#         """
#         .. todo::

#             WRITEME
#         """
#         self.input_space = space

#     def get_input_space(self):
#         """
#         .. todo::

#             WRITEME
#         """
#         if self.input_space is not None:
#             return self.input_space
#         raise ValueError("No input space was specified for this Block (%s). "
#                          "You can call set_input_space to correct that." %
#                          str(self))

#     def get_output_space(self):
#         """
#         .. todo::

#             WRITEME
#         """
#         return self.get_input_space()

if __name__=='__main__':
    # test 
    import cPickle

    with open('GTZAN_1024-40-fold-1_of_4.pkl') as f: config = cPickle.load(f)
    # D = TransformerDataset(
    #     raw = GTZAN_dataset2d(config),
    #     transformer = GTZAN_standardizer2d(config),
    #     space_preserving=False)

    n_components=80
    D = GTZAN_dataset2d(config, which_set='train', n_components=n_components, epsilon=3)

    conv_space   = Conv2DSpace(shape=(n_components,40), num_channels=1, axes=('b', 'c', 0, 1))
    feat_space   = VectorSpace(dim=n_components*40)    
    target_space = VectorSpace(dim=10)
    
    data_specs_conv = (CompositeSpace((conv_space,target_space)), ("features", "targets"))
    data_specs_feat = (CompositeSpace((feat_space,target_space)), ("features", "targets"))
    
    it_conv = D.iterator(mode='sequential', batch_size=10, data_specs=data_specs_conv)
    conv_batch = it_conv.next()

    it_feat = D.iterator(mode='sequential', batch_size=10, data_specs=data_specs_feat)
    feat_batch = it_feat.next()






