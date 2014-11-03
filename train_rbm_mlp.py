import sys, cPickle
from glob import glob
from pylearn2.train import Train
from pylearn2.utils import serial
from pylearn2.models.mlp import MLP, PretrainedLayer, Sigmoid, Softmax
from pylearn2.training_algorithms.sgd import SGD, LinearDecayOverEpoch
from pylearn2.training_algorithms.learning_rule import Momentum, MomentumAdjustor
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.datasets.transformer_dataset import TransformerDataset
from GTZAN_dataset import GTZAN_dataset, GTZAN_standardizer

def get_trainer(model, trainset, validset, save_path):
  
  monitoring  = dict(valid=validset, train=trainset)
  termination = MonitorBased(channel_name='valid_y_misclass', prop_decrease=.001, N=5)
  extensions  = [ MonitorBasedSaveBest(channel_name='valid_y_misclass', save_path=save_path),
                MomentumAdjustor(start=1, saturate=50, final_momentum=.9),
                LinearDecayOverEpoch(start=1, saturate=50, decay_factor=0.1) ]

  config = {
  'learning_rate': .01,
  'learning_rule': Momentum(0.5),
  'train_iteration_mode': 'shuffled_sequential',
  'batch_size': 1200,
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
  save_path = './saved/mlp_sigmoid' + ext

  rbm_models = glob('./saved/rbm_layer*'+ext)
  nlayers    = len(rbm_models)
  nvis       = 513
  nhid       = 50
  n_classes  = 10

  if 1: # use pretrained layer
    layers = []
    for i,rbm in enumerate(rbm_models):
      layers.append(PretrainedLayer(layer_name='h'+str(i), layer_content=serial.load( rbm )))

    layers.append(Softmax(n_classes=n_classes, layer_name='y', irange=5e-3))
    model = MLP(layers=layers, nvis=nvis)

  else: # manual copy
    layers = []
    for i in xrange(nlayers):
        layers.append(Sigmoid(layer_name='h'+str(i), dim=nhid, irange=5e-3))

    layers.append(Softmax(n_classes=n_classes, layer_name='y', irange=5e-3))
    model = MLP(layers=layers, nvis=nvis)

    for h, m in zip(rbm_models, model.layers[:-1]):
      layer = serial.load(h)
      m.set_weights( layer.get_weights() )
      if hasattr(layer,'bias_hid'):
        m.set_biases( layer.bias_hid.get_value() )
      elif hasattr(layer,'hidbias'):
        m.set_biases( layer.hidbias.get_value() )
      elif hasattr(layer,'get_biases'):
        m.set_biases( layer.get_biases() )

  trainset = TransformerDataset( raw=GTZAN_dataset(which_set='train', config=cfg), 
        transformer=GTZAN_standardizer(config=cfg) )

  validset = TransformerDataset( raw=GTZAN_dataset(which_set='valid', config=cfg), 
        transformer=GTZAN_standardizer(config=cfg) )

  train = get_trainer(model=model, trainset=trainset, validset=validset, save_path=save_path)
  train.main_loop()



