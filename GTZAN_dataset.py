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
from theano import config

import pdb

class GTZAN_framelevel_iterator(FiniteDatasetIterator):
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

        # !!! fancy indexing doesn't seem to work w/ pytables (data[next_index] doesn't work, but data[next_index,:] does) !!!
        rval = tuple(
            fn(data[next_index,:]) if fn else data[next_index,:] 
            for data, fn in safe_izip(self._raw_data, self._convert))
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval

class GTZAN_songlevel_iterator(FiniteDatasetIterator):
    '''
    Returns all data associated with a particular song from the dataset
    '''
    @functools.wraps(SubsetIterator.next)
    def next(self):
        
        n_frames_per_file = self._dataset.n_frames_per_file
        next_index = self._subset_iterator.next()        
        next_index = self._dataset.file_list[ next_index ] # !!! added line to iterate over different index set !!!
        next_index = np.hstack([i*n_frames_per_file + np.arange(n_frames_per_file) for i in next_index])        

        # !!! fancy indexing doesn't seem to work w/ pytables (data[next_index] doesn't work, but data[next_index,:] does) !!!
        rval = tuple(
            fn(data[next_index,:]) if fn else data[next_index,:] 
            for data, fn in safe_izip(self._raw_data, self._convert))
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval

class GTZAN_dataset(DenseDesignMatrixPyTables):

    def __init__(self, config, which_set='train'):

        keys = ['train', 'test', 'valid']
        assert which_set in keys

        # load h5file
        self.h5file  = tables.open_file( config['h5_file_name'], mode = "r" )
        data         = self.h5file.get_node('/', "Data")

        self.support = config[which_set]
        self.file_list = config[which_set+'_files']
        self.n_frames_per_file   = config['n_frames_per_file']
        self.n_frames_per_sample = config['n_frames_per_sample']

        super(GTZAN_dataset, self).__init__(X=data.X, y=data.y)

    def __del__(self):
        self.h5file.close()        

    # !!! copied from superclass so that we can return a GTZAN_iterator instead of FiniteDatasetIterator !!!
    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

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

        if 'songlevel-features' in sub_sources: # songlevel iterator
            return GTZAN_songlevel_iterator(self,
                              mode(len(self.file_list), batch_size, num_batches, rng),
                              data_specs=data_specs,
                              return_tuple=return_tuple,
                              convert=convert)
        else:
            return GTZAN_framelevel_iterator(self,
                                  mode(len(self.support), batch_size, num_batches, rng),
                                  data_specs=data_specs,
                                  return_tuple=return_tuple,
                                  convert=convert)

class GTZAN_standardizer(Block):

    def __init__(self, config):

        self._mean = np.array(config['mean'], dtype=np.float32)
        self._std  = np.array(config['std'], dtype=np.float32)
        self.input_space = VectorSpace(np.prod(self._mean.shape))

        super(GTZAN_standardizer, self).__init__()

    def __call__(self, batch):
        """
        .. todo::

            WRITEME
        """
        if self.input_space:
            self.input_space.validate(batch)

        return (batch - self._mean) / self._std

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME
        """
        self.input_space = space

    def get_input_space(self):
        """
        .. todo::

            WRITEME
        """
        if self.input_space is not None:
            return self.input_space
        raise ValueError("No input space was specified for this Block (%s). "
                         "You can call set_input_space to correct that." %
                         str(self))

    def get_output_space(self):
        """
        .. todo::

            WRITEME
        """
        return self.get_input_space()



if __name__=='__main__':
    # test 
    import cPickle

    with open('GTZAN_1024-fold-4_of_4.pkl') as f: config = cPickle.load(f)
    
    D = TransformerDataset(
        raw = GTZAN_dataset(config),
        transformer = GTZAN_standardizer(config))
    
    feat_space   = VectorSpace(dim=513)    
    target_space = VectorSpace(dim=10)
    
    data_specs_frame = (CompositeSpace((feat_space,target_space)), ("features", "targets"))
    data_specs_song = (CompositeSpace((feat_space,target_space)), ("songlevel-features", "targets"))

    framelevel_it = D.iterator(mode='sequential', batch_size=10, data_specs=data_specs_frame)
    frame_batch = framelevel_it.next()

    songlevel_it = D.iterator(mode='sequential', batch_size=1, data_specs=data_specs_song)    
    song_batch = songlevel_it.next()

