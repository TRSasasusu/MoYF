# coding: utf-8

import numpy as np
from . import common

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

    def learn(self, input_train, output_train, epoch_num, learning_rate=0.01):
        for epoch in range(epoch_num):
            last_outputs = []
            not_activated_outputs = []
            outputs = []
            
            for input_train_unit in input_train:
                not_activated_outputs_unit = []
                outputs_unit = []

                previous_output = input_train_unit
                for layer, weight, bias in zip(self.layers.layers, self.layers.weights, self.layers.biases):
                    not_activated_output = weight.dot(previous_output) + bias
                    not_activated_outputs_unit.append(not_activated_output)

                    previous_output = layer.activate(not_activated_output)
                    outputs_unit.append(previous_output)

                not_activated_outputs.append(np.array(not_activated_outputs_unit))
                outputs.append(np.array(outputs_unit))

                last_outputs.append(previous_output)

            # Backpropagation
            deltas = []
            dif_weights = []
            dif_biases = []

            # delta L(length - 1)
            deltas.append(output_train - last_outputs)

            length = len(self.layers.layers)
            for i in range(length):
                deltas.insert(0, common.dif(self.loss, not_activated_outputs[length - i - 2]) * (self.layers.weights[length - i - 1].dot(deltas[0])))

                dif_weights.insert(0, (1 / len(input_train)) * deltas[0].dot(outputs[length - i - 3].transpose()))
                dif_biases.insert(0, (1 / len(input_train)) * deltas[0].dot(np.ones(shape=(len(input_train), 1))))

            for i in range(length):
                weights[i] -= learning_rate * dif_weights[i]
                biases[i] -= learning_rate * dif_biases[i]

class Layer:
    def __init__(self, input_num=None, output_num, activate):
        self.input_num = input_num
        self.output_num = output_num
        self.activate = activate
