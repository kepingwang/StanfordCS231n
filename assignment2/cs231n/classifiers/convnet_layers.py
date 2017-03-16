import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNetLayer(object):

  def __init__(self):
    return
  
  def forward(self, X, mode):
    return X
  
  def backward(self, dout, mode):
    return dout

  def update_params(self, update_rule, lr):
    return

  def reg_loss(self):
    """
    Return the loss from parameter regularization.
    """
    return 0.0
  

class ConvLayer(ConvNetLayer):

  def __init__(self, W_size, weight_scale, reg, conv_param):
    """
    Input: 
    - W_size: tuple (F, C, HH, WW)
    - weight_scale: initial weight_scale
    - conv_param: A dictionary: with key 'stride' and 'pad'
    """
    F = W_size[0]
    self.W = np.random.standard_normal(W_size) * weight_scale
    self.b = np.zeros(F)
    self.reg = reg
    self.conv_param = conv_param
    self.config = {
      'W': {},
      'b': {}
    }

  def forward(self, X, mode='train'):
    out, self.cache = conv_forward_fast(X, self.W, self.b, self.conv_param)
    return out

  def backward(self, dout, mode='train'):
    dX, self.dW, self.db = conv_backward_fast(dout, self.cache)
    return dX

  def update_params(self, update_rule, lr):
    self.config['W']['learning_rate'] = lr
    self.config['b']['learning_rate'] = lr
    self.W, self.config['W'] = update_rule(self.W, self.dW, self.config['W'])
    self.b, self.config['b'] = update_rule(self.b, self.db, self.config['b'])

  def reg_loss(self):
    return 0.5 * self.reg * np.sum(self.W*self.W)

class MaxPoolLayer(ConvNetLayer):
  
  def __init__(self, pool_param):
    self.pool_param = pool_param
  
  def forward(self, X, mode='train'):
    out, self.cache = max_pool_forward_fast(X, self.pool_param)
    return out

  def backward(self, dout, mode='train'):
    return max_pool_backward_fast(dout, self.cache)

class AffineLayer(ConvNetLayer):

  def __init__(self, W_size, weight_scale, reg):
    self.W = np.random.standard_normal(W_size) * weight_scale
    self.b = np.zeros(W_size[1])
    self.reg = reg
    self.config = {
      'W': {},
      'b': {}
    }
  
  def forward(self, X, mode='train'):
    out, self.cache = affine_forward(X, self.W, self.b)
    return out

  def backward(self, dout, mode='train'):
    dX, self.dW, self.db = affine_backward(dout, self.cache)
    return dX

  def update_params(self, update_rule, lr):
    self.config['W']['learning_rate'] = lr
    self.config['b']['learning_rate'] = lr
    self.W, self.config['W'] = update_rule(self.W, self.dW, self.config['W'])
    self.b, self.config['b'] = update_rule(self.b, self.db, self.config['b'])

  def reg_loss(self):
    return 0.5 * self.reg * np.sum(self.W*self.W)

class ReluLayer(ConvNetLayer):

  def forward(self, X, mode='train'):
    out, self.cache = relu_forward(X)
    return out

  def backward(self, dout, mode='train'):
    dX = relu_backward(dout, self.cache)
    return dX

class BatchNormLayer(ConvNetLayer):

  def __init__(self, D, bn_param):
    self.gamma = np.ones(D)
    self.beta = np.zeros(D)
    self.bn_param = bn_param
    self.config = {
      'gamma': {},
      'beta': {}
    }

  def forward(self, X, mode='train'):
    self.bn_param['mode'] = mode
    out, self.cache = batchnorm_forward(X, self.gamma, self.beta, self.bn_param)
    return out

  def backward(self, dout, mode='train'):
    self.bn_param['mode'] = mode
    dX, self.dgamma, self.dbeta = batchnorm_backward_alt(dout, self.cache)
    return dX

  def update_params(self, update_rule, lr):
    self.config['gamma']['learning_rate'] = lr
    self.config['beta']['learning_rate'] = lr
    self.gamma, self.config['gamma'] = update_rule(self.gamma, self.dgamma, self.config['gamma'])
    self.beta, self.config['gamma']  = update_rule(self.beta,  self.dbeta,  self.config['beta'])


class SpatialBatchNormLayer(ConvNetLayer):

  def __init__(self, C, bn_param):
    self.gamma = np.ones(C)
    self.beta = np.zeros(C)
    self.bn_param = bn_param
    self.config = {
      'gamma': {},
      'beta': {}
    }

  def forward(self, X, mode='train'):
    self.bn_param['mode'] = mode
    out, self.cache = spatial_batchnorm_forward(X, self.gamma, self.beta, self.bn_param)
    return out

  def backward(self, dout, mode='train'):
    self.bn_param['mode'] = mode
    dX, self.dgamma, self.dbeta = spatial_batchnorm_backward(dout, self.cache)
    return dX

  def update_params(self, update_rule, lr):
    self.config['gamma']['learning_rate'] = lr
    self.config['beta']['learning_rate'] = lr
    self.gamma, self.config['gamma'] = update_rule(self.gamma, self.dgamma, self.config['gamma'])
    self.beta, self.config['beta']  = update_rule(self.beta,  self.dbeta,  self.config['beta'])

  def reg_loss(self):
    return 0.0

class DropOutLayer(ConvNetLayer):

  def __init__(self, dropout_param):
    """
    Input:
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.
    """
    self.dropout_param = dropout_param
  
  def forward(self, X, mode='train'):
    self.dropout_param['mode'] = mode
    out, self.cache = dropout_forward(X, self.dropout_param)

  def backward(self, dout, mode='train'):
    self.dropout_param['mode'] = mode
    return dropout_backward(dout, self.cache)
