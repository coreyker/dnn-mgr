import sys, re, cPickle
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
                #MomentumAdjustor(start=1, saturate=50, final_momentum=.9),
                LinearDecayOverEpoch(start=1, saturate=50, decay_factor=0.1)]

  config = {
  'learning_rate': .01,
  #'learning_rule': Momentum(0.5),
  'learning_rule': RMSProp(),
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

  _, layer_glob_pattern, save_path = sys.argv # e.g.,layer_glob_pattern = './saved/rbm_layer*-fold-1_of_4.cpu.pkl'
  pretrained_layers = sorted(glob(layer_glob_pattern))
  print pretrained_layers
  
  # get input model
  input_model = serial.load(pretrained_layers[0])  

  # get datasets for training and validation from pretrained input layer
  p = re.compile(r"which_set.*'(train)'")
  trainset_yaml = input_model.dataset_yaml_src
  validset_yaml = p.sub("which_set: 'valid'", trainset_yaml)

  trainset  = yaml_parse.load(trainset_yaml)
  validset  = yaml_parse.load(validset_yaml)  

  model     = get_mlp(input_model.nvis, pretrained_layers)
  trainset  = yaml_parse.load(trainset_yaml)
  validset  = yaml_parse.load(validset_yaml)
  trainer   = get_trainer(model=model, trainset=trainset, validset=validset, save_path=save_path)
  
  trainer.main_loop()



