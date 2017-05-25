import random
import numpy as np
import matplotlib.pyplot as plt
from moyf.ml import nn, common

def main():
    x_train = np.arange(-3, 3, 0.5)
    y_train = np.sin(x_train)
    y_train = [it + random.gauss(0, 0.2) for it in y_train]

    model = nn.Model(optimizer=nn.Sgd, loss=common.mse)

    model.layers.add(nn.Layer(input_num=1, output_num=5, activate=common.relu))
    model.layers.add(nn.Layer(output_num=4, activate=common.relu))
    model.layers.add(nn.Layer(output_num=1, activate=common.linear))

    model.learn(input_train=x_train, output_train=y_train, epoch_num=5, batch_num=6)

    x_result = np.arange(-3, 3, 0.1)
    y_result = model.result(input_result=x_result)

    plt.plot(x_train, y_train, 'o')
    plt.plot(x_result, y_result)

    plt.show()

if __name__ == '__main__':
    main()
