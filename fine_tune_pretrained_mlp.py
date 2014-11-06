import sys, cPickle
from glob import glob
from pylearn2.train import Train
from pylearn2.utils import serial
from pylearn2.models.mlp import MLP, PretrainedLayer, Sigmoid, Softmax
from pylearn2.training_algorithms.sgd import SGD, LinearDecayOverEpoch
from pylearn2.training_algorithms.learning_rule import Momentum, MomentumAdjustor
from pylearn2.termination_criteria import MonitorBased
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.datasets.transformer_dataset import TransformerDataset

import pylearn2.config.yaml_parse as yaml_parse
from GTZAN_dataset import GTZAN_dataset, GTZAN_standardizer

import pdb

def get_mlp(nvis, pretrained_layers):

  layer_yaml = []
  for i, m in enumerate(pretrained_layers):
    layer_yaml.append('''!obj:pylearn2.models.mlp.PretrainedLayer {
      layer_name : "%(layer_name)s",
      layer_content : !pkl: "%(layer_content)s"
      }''' % {'layer_name' : 'h'+str(i), 'layer_content' : m })

  layer_yaml.append('''!obj:pylearn2.models.mlp.Softmax {
    n_classes : 10,
    layer_name : "y",
    irange : .01
    }''')

  layer_yaml = ','.join(layer_yaml)

  model_yaml = '''!obj:pylearn2.models.mlp.MLP {
    nvis : %(nvis)i,
    layers : [%(layers)s]
    }''' % {'nvis' : nvis, 'layers' : layer_yaml}

  model = yaml_parse.load(model_yaml)
  return model

def get_trainer(model, trainset, validset, save_path):
  
  monitoring  = dict(valid=validset, train=trainset)
  termination = MonitorBased(channel_name='valid_y_misclass', prop_decrease=.001, N=5)
  extensions  = [MonitorBasedSaveBest(channel_name='valid_y_misclass', save_path=save_path),
                MomentumAdjustor(start=1, saturate=50, final_momentum=.9),
                LinearDecayOverEpoch(start=1, saturate=50, decay_factor=0.1)]

  config = {
  'learning_rate': .1,
  'learning_rule': Momentum(0.5),
  'train_iteration_mode': 'shuffled_sequential',
  'batch_size': 250,
  #'batches_per_iter' : 100,
  'monitoring_dataset': monitoring,
  'monitor_iteration_mode' : 'sequential',
  'termination_criterion' : termination,
  }

  return Train(model=model, 
      algorithm=SGD(**config),
      dataset=trainset,
      extensions=extensions)

if __name__=="__main__":

  fold_config = sys.argv[1] # e.g., GTZAN_1024-fold-1_of_4.pkl
  with open(fold_config) as f:
    cfg = cPickle.load(f)

  base = cfg['h5_file_name'].split('.h5')[0]
  ext  = fold_config.split(base)[1]
  ext  = ext.split('.pkl')[0] + '.cpu.pkl'  

  nvis = 513
  pretrained_layers = sorted(glob('./saved/rbm_layer*'+ext))
  
  trainset_yaml = '''!obj:pylearn2.datasets.transformer_dataset.TransformerDataset {
        raw : !obj:GTZAN_dataset.GTZAN_dataset {
            which_set : 'train',
            config : &fold !pkl: "%(fold_config)s"
        },
        transformer : !obj:GTZAN_dataset.GTZAN_standardizer {
            config : *fold
        }
    }''' % {'fold_config' : fold_config}

  validset_yaml = '''!obj:pylearn2.datasets.transformer_dataset.TransformerDataset {
        raw : !obj:GTZAN_dataset.GTZAN_dataset {
            which_set : 'valid',
            config : &fold !pkl: "%(fold_config)s"
        },
        transformer : !obj:GTZAN_dataset.GTZAN_standardizer {
            config : *fold
        }
    }''' % {'fold_config' : fold_config}


  save_path = './saved/mlp_sigmoid' + ext
  model     = get_mlp(nvis, pretrained_layers)
  trainset  = yaml_parse.load(trainset_yaml)
  validset  = yaml_parse.load(validset_yaml)
  trainer   = get_trainer(model=model, trainset=trainset, validset=validset, save_path=save_path)
  
  trainer.main_loop()



