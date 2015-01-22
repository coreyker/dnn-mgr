import sys, re, cPickle, argparse
from glob import glob
from pylearn2.train import Train
from pylearn2.utils import serial
from pylearn2.models.mlp import MLP, PretrainedLayer, Sigmoid, Softmax
from pylearn2.training_algorithms.sgd import SGD, LinearDecayOverEpoch
from pylearn2.training_algorithms.learning_rule import Momentum, MomentumAdjustor
from pylearn2.training_algorithms.learning_rule import RMSProp
from pylearn2.termination_criteria import MonitorBased
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.datasets.transformer_dataset import TransformerDataset
from audio_dataset import AudioDataset

import pylearn2.config.yaml_parse as yaml_parse

import pdb

def get_mlp(nvis, nclasses, pretrained_layers):

  layer_yaml = []
  for i, m in enumerate(pretrained_layers):
    layer_yaml.append('''!obj:pylearn2.models.mlp.PretrainedLayer {
      layer_name : "%(layer_name)s",
      layer_content : !pkl: "%(layer_content)s"
      }''' % {'layer_name' : 'h'+str(i), 'layer_content' : m })

  layer_yaml.append('''!obj:pylearn2.models.mlp.Softmax {
    n_classes : %(nclasses)i,
    layer_name : "y",
    irange : .01
    }''' % {'nclasses' : nclasses})

  layer_yaml = ','.join(layer_yaml)

  model_yaml = '''!obj:pylearn2.models.mlp.MLP {
    nvis : %(nvis)i,
    layers : [%(layers)s]
    }''' % {'nvis' : nvis, 'layers' : layer_yaml}

  model = yaml_parse.load(model_yaml)
  return model

def get_trainer(model, trainset, validset, save_path):
  
  monitoring  = dict(valid=validset, train=trainset)
  termination = MonitorBased(channel_name='valid_y_misclass', prop_decrease=.001, N=100)
  extensions  = [MonitorBasedSaveBest(channel_name='valid_y_misclass', save_path=save_path),
                #MomentumAdjustor(start=1, saturate=100, final_momentum=.9),
                LinearDecayOverEpoch(start=1, saturate=200, decay_factor=0.01)]

  config = {
  'learning_rate': .01,
  #'learning_rule': Momentum(0.5),
  'learning_rule': RMSProp(),
  'train_iteration_mode': 'shuffled_sequential',
  'batch_size': 1200,#250,
  #'batches_per_iter' : 100,
  'monitoring_dataset': monitoring,
  'monitor_iteration_mode' : 'shuffled_sequential',
  'termination_criterion' : termination,
  }

  return Train(model=model, 
      algorithm=SGD(**config),
      dataset=trainset,
      extensions=extensions)

if __name__=="__main__":

  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
  description='Script to pretrain the layers of a DNN.')

  parser.add_argument('fold_config', help='Path to dataset configuration file (generated with prepare_dataset.py)')
  parser.add_argument('--pretrained_layers', nargs='*', help='List of pretrained layers (sorted from input to output)')
  parser.add_argument('--save_file', help='Full path and for saving output model')
  args = parser.parse_args()
  
  trainset = yaml_parse.load(
    '''!obj:audio_dataset.AudioDataset {
             which_set : 'train',
             config : !pkl: "%(fold_config)s"
    }''' % {'fold_config' : args.fold_config} )

  validset = yaml_parse.load(
    '''!obj:audio_dataset.AudioDataset {
             which_set : 'valid',
             config : !pkl: "%(fold_config)s"
    }''' % {'fold_config' : args.fold_config} )


  testset = yaml_parse.load(
    '''!obj:audio_dataset.AudioDataset {
             which_set : 'test',
             config : !pkl: "%(fold_config)s"
    }''' % {'fold_config' : args.fold_config} )

  model   = get_mlp(nvis=trainset.X.shape[1], nclasses=trainset.y.shape[1], pretrained_layers=args.pretrained_layers)
  trainer = get_trainer(model=model, trainset=trainset, validset=validset, save_path=args.save_file)
  
  trainer.main_loop()



