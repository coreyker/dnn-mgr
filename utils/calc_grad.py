import numpy as np
from theano import function
from theano import tensor as T
import pylearn2
import audio_dataset
import pdb

'''
# !!! There is something wrong in the following gradient calculation !!!
# !!! (But we don't need it anyway, since theano can do the backprop calculation for us) !!!

# Calculate gradient of MLP w.r.t. input data
# (assumes rectified linear units and softmax output layer)

def calc_grad(X0, model, label):

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop( X, return_all=True )
    fprop = theano.function([X],Y)

    activations = fprop(X0)

    Wn = model.layers[-1].get_weights()
    bn = model.layers[-1].get_biases()
    Xn = activations[-1]

    # derivative of cost with respect to layer preceeding the softmax
    gradn = Wn[:,label] - Xn.dot(Wn.T) 

    pdb.set_trace()
    for n in xrange(len(model.layers)-2, 0, -1):
        Wn = model.layers[n].get_weights()
        bn = model.layers[n].get_biases()
        Xn_1 = activations[n-1]

        if type(model.layers[n]) is pylearn2.models.mlp.RectifiedLinear:
            dact = lambda x: x>0
        elif type(model.layers[n]) is pylearn2.models.mlp.Linear:
            dact = lambda x: x
        elif type(model.layers[n]) is audio_dataset.PreprocLayer:
            dact = lambda x: x

        gradn = (dact(Xn_1.dot(Wn)) * gradn).dot(Wn.T)

    return gradn
'''

# Create a simple model for testing
rng = np.random.RandomState(111)
epsilon = 1e-2
nvis = 10
nhid = 5
n_classes = 3

X0 = np.array(rng.randn(1,nvis), dtype=np.float32)
label = rng.randint(0,n_classes)

model = pylearn2.models.mlp.MLP(
    nvis=nvis,
    layers=[
        pylearn2.models.mlp.Linear(
            layer_name='pre',
            dim=nvis,
            irange=1.
            ),
        pylearn2.models.mlp.RectifiedLinear(
            layer_name='h0',
            dim=nhid,
            irange=1.),
        pylearn2.models.mlp.RectifiedLinear(
            layer_name='h1',
            dim=nhid,
            irange=1.),
        pylearn2.models.mlp.RectifiedLinear(
            layer_name='h2',
            dim=nhid,
            irange=1.),
        pylearn2.models.mlp.Softmax(
            n_classes=n_classes,
            layer_name='y',
            irange=1.)
        ])

# Numerical computation of gradients
X = model.get_input_space().make_theano_batch()
Y = model.fprop( X )
fprop = function([X],Y)

dX_num = np.zeros(X0.shape)
for i in range(nvis):
    tmp = np.copy(X0[:,i])
    X0[:,i] = tmp + epsilon
    Y_plus  = -np.log(fprop(X0)[:,label])
    
    X0[:,i] = tmp - epsilon
    Y_minus = -np.log(fprop(X0)[:,label])

    X0[:,i] = tmp
    dX_num[:,i] = (Y_plus - Y_minus) / (2*epsilon)

# Computation of gradients using Theano
n_examples = X0.shape[0]
label_vec = T.vector('label_vec')
cost  = model.cost(label_vec, model.fprop(X))
dCost = T.grad(cost * n_examples, X) 
f = function([X, label_vec], dCost)

one_hot = np.zeros(n_classes, dtype=np.float32)
one_hot[label] = 1

dX_est = f(X0, one_hot) #dX_est = calc_grad(X0, model, label)

delta = dX_num - dX_est
# Print results
print 'Numerical gradient:', dX_num
print 'Theano gradient:', dX_est
print 'Absolute difference:', np.abs(delta)
print '2-norm of difference', np.linalg.norm(delta)

