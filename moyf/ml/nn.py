# coding: utf-8

class Model:
    class Layers:
        def __init__(self):
            self.layers = []
            self.weights = []
            self.biases = []

        def add(self, layer):
            if len(self.layers) == 0 and layer.input_num is None:
                raise Exception('There is not input_num!')

            layer.input_num = self.layers[-1].output_num
            self.layers.append(layer)

    def __init__(self, optimizer=None, loss=None):
        self.optimizer = optimizer
        self.loss = loss
        self.layers = Layers()

class Layer:
    def __init__(self, input_num=None, output_num, activate):
        self.input_num = input_num
        self.output_num = output_num
        self.activate = activate
