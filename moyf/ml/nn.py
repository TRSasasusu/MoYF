# coding: utf-8

import numpy as np

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
            self.weights.append(np.random.normal(0, 1, size=(layer.output_num, layer.input_num)))
            self.biases.append(np.zeros(shape=(layer.output_num, layer.input_num)))

    def __init__(self, optimizer=None, loss=None):
        self.optimizer = optimizer
        self.loss = loss
        self.layers = Layers()

    def learn(self, input_train, output_train, epoch_num):
        for epoch in range(epoch_num):
            outputs = []
            for input_train_unit in input_train:
                previous_output = input_train_unit
                for layer, weight, bias in zip(self.layers, self.weights, self.biases):
                    not_activated_output = weight.dot(previous_output) + bias
                    previous_output = layer.activate(not_activated_output)
                outputs.append(previous_output)

            

class Layer:
    def __init__(self, input_num=None, output_num, activate):
        self.input_num = input_num
        self.output_num = output_num
        self.activate = activate
