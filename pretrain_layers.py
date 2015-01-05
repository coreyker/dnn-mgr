import os, sys, cPickle, argparse
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.ebm_estimation import SML
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.termination_criteria import MonitorBased, ChannelTarget, EpochCounter
from pylearn2.training_algorithms.learning_rule import RMSProp
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError

import pylearn2.config.yaml_parse as yaml_parse
from audio_dataset import AudioDataset, PreprocLayer
import pdb

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
        corruptor : !obj:pylearn2.corruption.BinomialCorruptor { corruption_level : .1 },
        act_enc : 'sigmoid',
        act_dec : null  
    }''' % {'nvis' : nvis, 'nhid': nhid}  

    model = yaml_parse.load(model_yaml)
    return model

def get_rbm_trainer(model, dataset, save_path, epochs=5):
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
    'termination_criterion' : EpochCounter(max_epochs=epochs),
    }
    
    return Train(model=model, 
        algorithm=SGD(**config),
        dataset=dataset,    
        save_path=save_path, 
        save_freq=1
        )#, extensions=extensions)

def get_ae_trainer(model, dataset, save_path, epochs=5):
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
    'termination_criterion' : EpochCounter(max_epochs=epochs),
    }

    return Train(model=model, 
        algorithm=SGD(**config),
        dataset=dataset,    
        save_path=save_path, 
        save_freq=1
        )#, extensions=extensions)

if __name__=="__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    description='Script to pretrain the layers of a DNN.')

    parser.add_argument('fold_config', help='Path to dataset configuration file (generated with prepare_dataset.py)')
    parser.add_argument('--arch', nargs='*', type=int, help='Architecture: nvis nhid1 nhid2 ...')
    parser.add_argument('--epochs', type=int, help='Number of training epochs per layer')
    parser.add_argument('--save_prefix', help='Full path and prefix for saving output models')
    parser.add_argument('--use_autoencoder', action='store_true')
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = 5

    arch = [(i,j) for i,j in zip(args.arch[:-1], args.arch[1:])]

    with open(args.fold_config) as f: 
        config = cPickle.load(f)

    preproc_layer = PreprocLayer(config=config, proc_type='standardize')

    dataset = TransformerDataset(
        raw=AudioDataset(which_set='train', config=config),
        transformer=preproc_layer.layer_content
        )

    # transformer_yaml = '''!obj:pylearn2.datasets.transformer_dataset.TransformerDataset {
    #     raw : %(raw)s,
    #     transformer : %(transformer)s
    # }'''
    #
    # dataset_yaml = transformer_yaml % {
    #     'raw' : '''!obj:audio_dataset.AudioDataset {
    #         which_set : 'train',
    #         config : !pkl: "%(fold_config)s"
    #     }''' % {'fold_config' : args.fold_config},
    #     'transformer' : '''!obj:pylearn2.models.mlp.MLP {
    #         nvis : %(nvis)i,
    #         layers : 
    #         [
    #             !obj:audio_dataset.PreprocLayer {
    #                 config : !pkl: "%(fold_config)s",
    #                 proc_type : 'standardize'
    #             } 
    #         ]
    #     }''' % {'nvis' : args.arch[0], 'fold_config' : args.fold_config }
    # }

    for i,(v,h) in enumerate(arch):        

        if not args.use_autoencoder:
            print 'Pretraining layer %d with RBM' % i

            if i==0:
                model = get_grbm(v,h)
            else:
                model = get_rbm(v,h)

            save_path = args.save_prefix+ 'RBM_L{}.pkl'.format(i+1)
            trainer   = get_rbm_trainer(model=model, dataset=dataset, save_path=save_path, epochs=args.epochs)
        else:
            print 'Pretraining layer %d with AE' % i

            model     = get_ae(v,h)
            save_path = args.save_prefix + 'AE_L{}.pkl'.format(i+1)
            trainer   = get_ae_trainer(model=model, dataset=dataset, save_path=save_path, epochs=args.epochs)

        trainer.main_loop()

        dataset = TransformerDataset(raw=dataset, transformer=model)

        # dataset_yaml = transformer_yaml % {'raw' : dataset_yaml, 'transformer' : '!pkl: %s' % save_path}
        # dataset = yaml_parse.load( dataset_yaml )


