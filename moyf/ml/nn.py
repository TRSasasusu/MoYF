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
#                if not_activated_outputs is None:
#                    not_activated_outputs = np.array([weight.dot(previous_output) + bias.dot(np.ones(shape=(1, len(input_train))))])
#                else:
#                    import ipdb; ipdb.set_trace()
#                    np.vstack((not_activated_outputs, [weight.dot(previous_output) + bias.dot(np.ones(shape=(1, len(input_train))))]))

                outputs.append(np.array(layer.activate(not_activated_outputs[-1])))
#                if outputs is None:
#                    outputs = np.array([layer.activate(not_activated_outputs[-1])])
#                else:
#                    np.vstack((outputs, [layer.activate(not_activated_outputs[-1])]))

                previous_output = outputs[-1]

            # last_outputs: array([[], [], ..., []])
            last_outputs = previous_output
            '''















#            last_outputs = np.empty(shape=(1, output_trai
            last_outputs = None
            not_activated_outputs = [[] for i in self.layers.layers]
            outputs = [[] for i in self.layers.layers]
            
#            import ipdb; ipdb.set_trace()
            for input_train_unit in input_train:
                previous_output = input_train_unit
                for layer, weight, bias, i in zip(self.layers.layers, self.layers.weights, self.layers.biases, range(len(self.layers.layers))):
#                    import ipdb; ipdb.set_trace()
                    not_activated_output = weight.dot(previous_output.reshape(len(previous_output), 1)) + bias

                    previous_output = layer.activate(not_activated_output.transpose()[0]).transpose()
#                    import ipdb; ipdb.set_trace()

                    if len(not_activated_outputs[i]) == 0:
                        not_activated_outputs[i].append(not_activated_output)
                        outputs[i].append(previous_output.reshape(len(previous_output), 1))
                    else:
                        not_activated_outputs[i] = [np.hstack((not_activated_outputs[i][0], not_activated_output))]
                        outputs[i] = [np.hstack((outputs[i][0], previous_output.reshape(len(previous_output), 1)))]

                if last_outputs is None:
                    last_outputs = previous_output
                else:
                    last_output.vstack(previous_output)
            '''
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
#                for k in range(len(not_activated_outputs[length - i - 2][0])):
#                    not_activated_outputs[length - i - 2][0][k] = common.dif(self.layers.layers[length - i - 2].activate, not_activated_outputs[length - i - 2][0][k])
                not_activated_outputs[length - i - 2] = common.dif(self.layers.layers[length - i - 2].activate, not_activated_outputs[length - i - 2])
                deltas.insert(0, not_activated_outputs[length - i - 2] * (self.layers.weights[length - i - 1].transpose().dot(deltas[0])))

                if length - i - 3 < 0:
                    dif_weights.insert(0, (1 / len(input_train)) * deltas[0].dot(input_train))
                else:
#                    import ipdb; ipdb.set_trace()
                    dif_weights.insert(0, (1 / len(input_train)) * deltas[0].dot(outputs[length - i - 3].transpose()))
                dif_biases.insert(0, (1 / len(input_train)) * deltas[0].dot(np.ones(shape=(len(input_train), 1))))

                if length - i - 2 == 0:
                    break

            for i in range(length):
#                import ipdb; ipdb.set_trace()
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
        last_outputs = []

        for input_result_unit in input_result:
            previous_output = np.array([input_result_unit])
            for layer, weight, bias in zip(self.layers.layers, self.layers.weights, self.layers.biases):
#                import ipdb; ipdb.set_trace()
                not_activated_output = weight.dot(previous_output.reshape(len(previous_output), 1)) + bias
                previous_output = layer.activate(not_activated_output.transpose()[0]).transpose()
            last_outputs.append(previous_output)

        return np.array(last_outputs)

class Layer:
    def __init__(self, activate, output_num, input_num=None):
        self.input_num = input_num
        self.output_num = output_num
        self.activate = activate
