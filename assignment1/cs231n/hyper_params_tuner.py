import random
from classifiers.neural_net import TwoLayerNet

class HyperParamsTuner():
    """
    Hyper parameters tuner for TwoLayerNet.
    """
    
    def __init__(self, hyper_params_range, X_train, y_train, X_val, y_val, num_classes):
        self.mapping = dict(hyper_params_range)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_size = X_train.shape[1]
        self.num_classes = num_classes
        self.best_param = None
        self.best_net = None
        self.best_val_acc = -1
        self.results = []
        
    def get_results(self):
        return self.results
    
    def get_best_result(self):
        return (self.best_param, self.best_net, self.best_val_acc)
        
    def set_hyper_params_range(self, hyper_params_range):
        self.mapping = dict(hyper_params_range)
    
    def next_param(self):
        param = {}
        for key in self.mapping:
            param[key] = self.mapping[key](random.random())
        return param
    
    def __evaluate(self, param):
        print 'parameters: ', param
        net = TwoLayerNet(self.input_size, param['hidden_size'], self.num_classes)
        # Train the network
        stats = net.train(self.X_train, self.y_train, self.X_val, self.y_val,
                    num_iters=param['num_iters'], batch_size=param['batch_size'],
                    learning_rate=param['learning_rate'], learning_rate_decay=0.95,
                    reg=param['reg'], verbose=True)
        # Predict on the validation set
        trn_acc = (net.predict(self.X_train) == self.y_train).mean()
        val_acc = (net.predict(self.X_val) == self.y_val).mean()
        print 'Validation accuracy: ', val_acc
        result = {}
        result['params'] = param
        result['trn_acc'] = trn_acc
        result['val_acc'] = val_acc
        self.results.append(result)
        if val_acc > self.best_val_acc:
            self.best_param = param
            self.best_net = net
            self.best_val_acc = val_acc
            
    def evaluate(self):
        param = self.next_param()
        self.__evaluate(param)
