import copy
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *
from cs231n.classifiers.convnet_layers import *
from cs231n.solver import Solver
from cs231n import optim


class ConvNet(object):
  """
  A custom convNet. Linear architecture.
  """

  def __init__(self, input_dim=3*32*32, num_classes=10, 
               weight_scale=1e-3, reg=0.0, use_batchnorm=True, dropout=0):
    """
    Initialize a ConvNet object. The specific layer configuration
    is to be added later. By default the end is svm
    """
    self.layers = []
    self.weight_scale = weight_scale
    self.reg = reg
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.bn_param = {
      'eps': 1e-5,
      'momentum': 0.9
    }
    self.dropout_param = {
      'p': dropout
    }


  def add_conv_relu_layer(self, W_size, conv_param):
    self.layers.append(ConvLayer(W_size, self.weight_scale, self.reg, conv_param))
    if self.use_batchnorm:
      self.layers.append(SpatialBatchNormLayer(W_size[0], dict(self.bn_param)))
    self.layers.append(ReluLayer())
    if self.use_dropout:
      self.layers.append(DropOutLayer(dict(self.dropout_param)))

  def add_affine_layer(self, W_size):
    self.layers.append(AffineLayer(W_size, self.weight_scale, self.reg))
    if self.use_batchnorm:
      self.layers.append(BatchNormLayer(W_size[1], dict(self.bn_param)))

  def add_affine_relu_layer(self, W_size):
    self.layers.append(AffineLayer(W_size, self.weight_scale, self.reg))
    if self.use_batchnorm:
      self.layers.append(BatchNormLayer(W_size[1], dict(self.bn_param)))
    self.layers.append(ReluLayer())
    if self.use_dropout:
      self.layers.append(DropOutLayer(dict(self.dropout_param)))

  def add_max_pool_layer(self, pool_param):
    self.layers.append(MaxPoolLayer(pool_param))

  def loss(self, X, y=None):
    mode = 'test' if y is None else 'train'
    out = X
    for layer in self.layers:
      out = layer.forward(out, mode)
      # print type(layer).__name__

    if mode == 'test':
      return out

    loss, dout = softmax_loss(out, y) # hard-coded softmax loss
    for layer in self.layers:
      loss += layer.reg_loss()

    for layer in reversed(self.layers):
      dout = layer.backward(dout, mode)

    return loss

  def update_params(self, update_rule, learning_rate):
    for layer in self.layers:
      layer.update_params(update_rule, learning_rate)



class ConvNetSolver(Solver):
  def __init__(self, model, data, **kwargs):
    """
    Construct a new Solver instance.
    
    Required arguments:
    - model: A model object conforming to the API described above
    - data: A dictionary of training and validation data with the following:
      'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
      'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
      'y_train': Array of shape (N_train,) giving labels for training images
      'y_val': Array of shape (N_val,) giving labels for validation images
      
    Optional arguments:
    - update_rule: A string giving the name of an update rule in optim.py.
      Default is 'sgd'.
    - learning_rate: learning rate
    - lr_decay: A scalar for learning rate decay; after each epoch the learning
      rate is multiplied by this value.
    - batch_size: Size of minibatches used to compute loss and gradient during
      training.
    - num_epochs: The number of epochs to run for during training.
    - print_every: Integer; training losses will be printed every print_every
      iterations.
    - verbose: Boolean; if set to false then no output will be printed during
      training.
    """
    self.model = model
    self.X_train = data['X_train']
    self.y_train = data['y_train']
    self.X_val = data['X_val']
    self.y_val = data['y_val']
    
    # Unpack keyword arguments
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.learning_rate = kwargs.pop('learning_rate', 1e-3)
    self.lr_decay = kwargs.pop('lr_decay', 1.0)
    self.batch_size = kwargs.pop('batch_size', 100)
    self.num_epochs = kwargs.pop('num_epochs', 10)

    self.print_every = kwargs.pop('print_every', 10)
    self.verbose = kwargs.pop('verbose', True)

    # Throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
      extra = ', '.join('"%s"' % k for k in kwargs.keys())
      raise ValueError('Unrecognized arguments %s' % extra)

    # Make sure the update rule exists, then replace the string
    # name with the actual function
    if not hasattr(optim, self.update_rule):
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)
    self.update_rule = getattr(optim, self.update_rule)

    self._reset()


  def _reset(self):
    """
    Set up some book-keeping variables for optimization. Don't call this
    manually.
    """
    # Set up some variables for book-keeping
    self.epoch = 0
    self.best_val_acc = 0
    self.best_model = self.model
    self.loss_history = []
    self.train_acc_history = []
    self.val_acc_history = []


  def _step(self):
    """
    Make a single gradient update. This is called by train() and should not
    be called manually.
    """
    # Make a minibatch of training data
    num_train = self.X_train.shape[0]
    batch_mask = np.random.choice(num_train, self.batch_size)
    X_batch = self.X_train[batch_mask]
    y_batch = self.y_train[batch_mask]

    # Compute loss and gradient
    loss = self.model.loss(X_batch, y_batch)
    self.loss_history.append(loss)

    # Perform a parameter update
    self.model.update_params(self.update_rule, self.learning_rate)

  def train(self):
    """
    Run optimization to train the model.
    """
    num_train = self.X_train.shape[0]
    iterations_per_epoch = max(num_train / self.batch_size, 1)
    num_iterations = int(self.num_epochs * iterations_per_epoch)

    for t in xrange(num_iterations):
      self._step()

      # Maybe print training loss
      if self.verbose and t % self.print_every == 0:
        print '(Iteration %d / %d) loss: %f' % (
               t + 1, num_iterations, self.loss_history[-1])

      # At the end of every epoch, increment the epoch counter and decay the
      # learning rate.
      epoch_end = (t + 1) % iterations_per_epoch == 0
      if epoch_end:
        self.epoch += 1
        self.learning_rate *= self.lr_decay

      # Check train and val accuracy on the first iteration, the last
      # iteration, and at the end of each epoch.
      first_it = (t == 0)
      last_it = (t == num_iterations + 1)
      if first_it or last_it or epoch_end:
        train_acc = self.check_accuracy(self.X_train, self.y_train,
                                        num_samples=1000)
        val_acc = self.check_accuracy(self.X_val, self.y_val)
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)

        if self.verbose:
          print '(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                 self.epoch, self.num_epochs, train_acc, val_acc)

        # Keep track of the best model
        if val_acc > self.best_val_acc:
          self.best_val_acc = val_acc
          self.best_model = copy.deepcopy(self.model)

    # At the end of training swap the best params into the model
    self.model = self.best_model



