import sys, cPickle
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.ebm_estimation import SML
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.termination_criteria import MonitorBased, ChannelTarget, EpochCounter
from pylearn2.training_algorithms.learning_rule import RMSProp
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError

import pylearn2.config.yaml_parse as yaml_parse
from GTZAN_dataset import GTZAN_dataset, GTZAN_standardizer

import pdb

MAX_EPOCHS_UNSUPERVISED = 5
USE_RBM_PRETRAIN = True

'''
(Although it may be more complicated) We build our models and dataset using yaml in order to keep a record of how things were built
''' 

def get_grbm(nvis, nhid):

    model_yaml = '''!obj:pylearn2.models.rbm.GaussianBinaryRBM {
        nvis : %(nvis)i,
        nhid : %(nhid)i,
        irange : .1,
        energy_function_class : !obj:pylearn2.energy_functions.rbm_energy.grbm_type_1 {},        
        init_sigma : 1.,
        init_bias_hid : 0,
        mean_vis : True    
    }''' % {'nvis' : nvis, 'nhid': nhid}    

    model = yaml_parse.load(model_yaml)
    return model

def get_rbm(nvis, nhid):

    model_yaml = '''!obj:pylearn2.models.rbm.RBM {
        nvis : %(nvis)i,
        nhid : %(nhid)i,
        irange : .1
    }''' % {'nvis' : nvis, 'nhid': nhid}  

    model = yaml_parse.load(model_yaml)
    return model

def get_ae(nvis, nhid):

    model_yaml = '''!obj:pylearn2.models.autoencoder.DenoisingAutoencoder {
        nvis : %(nvis)i,
        nhid : %(nhid)i,
        irange : .1,
        corruptor : !obj:pylearn2.corruption.BinomialCorruptor(corruption_level=0.1),
        act_enc : 'sigmoid',
        act_dec : None  
    }''' % {'nvis' : nvis, 'nhid': nhid}  

    model = yaml_parse.load(model_yaml)
    return model

def get_rbm_trainer(model, dataset, save_path):
    """
    A Restricted Boltzmann Machine (RBM) trainer    
    """

    config = {
    'learning_rate': 1e-2,
    'train_iteration_mode': 'shuffled_sequential',
    'batch_size': 250,
    #'batches_per_iter' : 100,
    'learning_rule': RMSProp(),
    'monitoring_dataset': dataset,
    'cost' : SML(250, 1),
    'termination_criterion' : EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED),
    }
    
    return Train(model=model, 
        algorithm=SGD(**config),
        dataset=dataset,    
        save_path=save_path, 
        save_freq=1
        )#, extensions=extensions)

def get_ae_trainer(model, dataset, save_path):
    """
    An Autoencoder (AE) trainer    
    """

    config = {
    'learning_rate': 1e-2,
    'train_iteration_mode': 'shuffled_sequential',
    'batch_size': 250,
    #'batches_per_iter' : 2000,
    'learning_rule': RMSProp(),
    'monitoring_dataset': dataset,
    'cost' : MeanSquaredReconstructionError(),
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
    arch   = [[nvis, nhid], [nhid, nhid], [nhid, nhid]]

    transformer_yaml = '''!obj:pylearn2.datasets.transformer_dataset.TransformerDataset {
        raw : %(raw)s,
        transformer : %(transformer)s
    }'''

    dataset_yaml = transformer_yaml % {
        'raw' : '''!obj:GTZAN_dataset.GTZAN_dataset {
            which_set : 'train',
            config : !pkl: "%(fold_config)s"
            }''' % {'fold_config' : fold_config},
        'transformer' : '''!obj:GTZAN_dataset.GTZAN_standardizer {
            config : !pkl: "%(fold_config)s"
            }''' % {'fold_config' : fold_config},
        }

    for i,(v,h) in enumerate(arch):

        
        dataset = yaml_parse.load( dataset_yaml )

        if USE_RBM_PRETRAIN:
            if i==0:
                model = get_grbm(v,h)
            else:
                model = get_rbm(v,h)

            save_path = './saved/rbm_layer%d%s' % (i+1, ext)
            trainer   = get_rbm_trainer(model=model, dataset=dataset, save_path=save_path)
        else:
            model     = get_ae(v,h)
            save_path = './saved/ae_layer%d%s' % (i+1, ext)
            trainer   = get_ae_trainer(model=model, dataset=dataset, save_path=save_path)

        trainer.main_loop()

        dataset_yaml = transformer_yaml % {'raw' : dataset_yaml, 'transformer' : '!pkl: %s' % save_path}



