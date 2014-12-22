import numpy as np
import argparse
from theano import function
from theano import tensor as T
from pylearn2.utils import serial
from audio_dataset import AudioDataset
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace, IndexSpace
import pylearn2.config.yaml_parse as yaml_parse
import pdb

# find adversarial examples
def generate_adversary(model, X0, label, epsilon=.25):
    '''
    model - dnn model
    X0 - an example that the model classifies correctly
    label - an incorrect label
    '''

    # convert integer label into one-hot vector
    n_classes = model.get_output_space().dim
    one_hot = np.zeros(n_classes, dtype=np.float32)
    one_hot[label] = 1

    # Computation of gradients using Theano
    X = model.get_input_space().make_theano_batch()
    label_vec = T.vector('label_vec')
    cost  = model.cost(label_vec, model.fprop(X))
    dCost = T.grad(cost, X) 
    f = function([X, label_vec], dCost)

    delta = f(X0, one_hot)
    X_adv = X0 + epsilon*delta

    return X_adv

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    description='''Script to find/test adversarial examples with a dnn''')
    parser.add_argument('model', help='')
    args = parser.parse_args()

    model = serial.load(args.model)  
    dataset = yaml_parse.load(model.dataset_yaml_src)

    test_file = 'classical/classical.00000.wav'
    offset, nframes, label, target = dataset.file_index[test_file]

    X     = model.get_input_space().make_theano_batch()
    Y     = model.fprop( X )
    fprop = function([X],Y)

    X0 = dataset.X[offset:offset+nframes,:]
    Y0 = fprop(X0)

    # find a 'correctly' classified frame
    for x, y in zip(X0, Y0):
        if np.argmax(y)==target:
            print 'Found correctly classified frame (confidence: {})'.format(y[target])
            break

    print 'Attempting to find adversarial example'
    decoy = np.random.choice( np.setdiff1d(dataset.targets, [target]) )
    X_adv = generate_adversary(model, x, decoy)

    print 'Testing if adversarial example is mis-classified'
    y = fprop(X_adv)
    if np.argmax(y) is not target:
        print 'Adversarial example mis-classified as {} (should be {})'.format(np.argmax(y), target)
    else:
        print 'Unsuccessful at generating an adversary'

