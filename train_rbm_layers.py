import sys, cPickle
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.models.rbm import GaussianBinaryRBM, RBM
from pylearn2.costs.ebm_estimation import SMD
from pylearn2.corruption import GaussianCorruptor
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.termination_criteria import MonitorBased, ChannelTarget, EpochCounter
from pylearn2.base import StackedBlocks
from GTZAN_dataset import GTZAN_dataset, GTZAN_standardizer

MAX_EPOCHS_UNSUPERVISED = 2

def get_rbm_trainer(model, dataset, save_path):

    config = {
    'learning_rate': 1e-2,
    'train_iteration_mode': 'shuffled_sequential',
    'batch_size': 1200,
    #'batches_per_iter' : 100,
    'monitoring_dataset': dataset,
    'cost' : SMD(corruptor=GaussianCorruptor(stdev=0.4)),
    'termination_criterion' : EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED),
    }
    
    return Train(model=model, 
        algorithm=SGD(**config),
        dataset=dataset,
        save_path=save_path, 
        save_freq=1
        )#, extensions=extensions)

if __name__=="__main__":

    fold_config = sys.argv[1] # e.g., GTZAN_1024-fold-1_of_4.pkl
    with open(fold_config) as f:
        cfg = cPickle.load(f)

    base  = cfg['h5_file_name'].split('.h5')[0]
    ext   = fold_config.split(base)[1]
    
    nvis   = 513
    nhid   = 50
    arch   = [[nhid, nhid], [nhid, nhid]]
    irange = 0.1

    layers = [ RBM(nvis=nvis, nhid=nhid, irange=irange) ]

    datasets = [ TransformerDataset( raw=GTZAN_dataset(which_set='train', config=cfg), 
        transformer=GTZAN_standardizer(config=cfg) )]

    for v,h in arch:
        dset = TransformerDataset( raw=datasets[-1], transformer=layers[-1])
        layers.append( RBM(nvis=v, nhid=h, irange=irange))
        datasets.append(dset)

    for i, (layer, dset) in enumerate(zip(layers, datasets)):
        save_path = './saved/rbm%s_layer%d.pkl' % (ext, i+1)
        train = get_rbm_trainer(model=layer, dataset=dset, save_path=save_path)
        train.main_loop()

