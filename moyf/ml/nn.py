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
            if layer.input_num is None:
                if len(self.layers) == 0:
                    raise Exception('There is not input_num!')
                else:
                    layer.input_num = self.layers[-1].output_num

            self.layers.append(layer)
            self.weights.append(np.random.normal(0, 1, size=(layer.output_num, layer.input_num)))
            self.biases.append(np.zeros(shape=(layer.output_num, 1)))

    def __init__(self, optimizer=None, loss=None):
        #self.optimizer = optimizer TODO
        self.loss = loss
        self.layers = Model.Layers()

    def learn(self, input_train, output_train, epoch_num, learning_rate=0.01):
#        import ipdb; ipdb.set_trace()
        for epoch in range(epoch_num):
            # not_activated_outputs, outputs: [array([[], [], ..., []]), array([[], [], ..., []]), ..., array([[], [], ..., []])]
            not_activated_outputs = []
            outputs = []

            # previous_output: array([[], [], ..., []])
            previous_output = input_train.T

            for layer, weight, bias in zip(self.layers.layers, self.layers.weights, self.layers.biases):
                not_activated_outputs.append(np.array(weight.dot(previous_output) + bias.dot(np.ones(shape=(1, len(input_train))))))
                outputs.append(np.array(layer.activate(not_activated_outputs[-1])))

                previous_output = outputs[-1]

            # last_outputs: array([[], [], ..., []])
            last_outputs = previous_output

            # Backpropagation
            deltas = []
            dif_weights = []
            dif_biases = []

            # delta L(length - 1)
            length = len(self.layers.layers)
            deltas.append(output_train.T - last_outputs)
            dif_weights.insert(0, (1 / len(input_train)) * deltas[0].dot(outputs[length - 2].transpose()))
            dif_biases.insert(0, (1 / len(input_train)) * deltas[0].dot(np.ones(shape=(len(input_train), 1))))

            for i in range(length):
                not_activated_outputs[length - i - 2] = common.dif(self.layers.layers[length - i - 2].activate, not_activated_outputs[length - i - 2])
                deltas.insert(0, not_activated_outputs[length - i - 2] * (self.layers.weights[length - i - 1].transpose().dot(deltas[0])))

                if length - i - 3 < 0:
                    dif_weights.insert(0, (1 / len(input_train)) * deltas[0].dot(input_train))
                else:
                    dif_weights.insert(0, (1 / len(input_train)) * deltas[0].dot(outputs[length - i - 3].transpose()))
                dif_biases.insert(0, (1 / len(input_train)) * deltas[0].dot(np.ones(shape=(len(input_train), 1))))

                if length - i - 2 == 0:
                    break

            for i in range(length):
                self.layers.weights[i] += learning_rate * dif_weights[i]
                self.layers.biases[i] += learning_rate * dif_biases[i]

            print("E = %s" % self.loss(output_train=output_train, last_outputs=last_outputs.T))
#            print("dif_weights:")
#            print(dif_weights)
            print("weights:")
            print(self.layers.weights)
            print("biases:")
            print(self.layers.biases)

    def result(self, input_result):
        previous_output = input_result.T

        for layer, weight, bias in zip(self.layers.layers, self.layers.weights, self.layers.biases):
            not_activated_output = np.array(weight.dot(previous_output) + bias.dot(np.ones(shape=(1, len(input_result)))))
            previous_output = np.array(layer.activate(not_activated_output))

        return previous_output.T

class Layer:
    def __init__(self, activate, output_num, input_num=None):
        self.input_num = input_num
        self.output_num = output_num
        self.activate = activate
