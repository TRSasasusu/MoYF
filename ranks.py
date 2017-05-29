# coding: utf-8

import numpy as np
from moyf.ml import nn, common

def main():
    # For example, scores of ranks.
    input_data = np.array([np.random.randint(1, 6, 8) for i in range(1020)])

    input_train = input_data[:1000]
    input_result = input_data[1000:]
    output_train = np.array([[1, 0, 0] if x.sum() > 30 else ([0, 1, 0] if x.sum() > 20 else [0, 0, 1]) for x in input_train])

    model = nn.Model(loss=common.cross_entropy)

    model.layers.add(nn.Layer(input_num=8, output_num=60, activate=common.relu))
    model.layers.add(nn.Layer(output_num=20, activate=common.relu))
    model.layers.add(nn.Layer(output_num=3, activate=common.softmax))

    model.learn(input_train=input_train, output_train=output_train, epoch_num=1000)

    output_result = model.result(input_result=input_result)

    for i_unit, o_unit in zip(input_result, output_result):
        print(i_unit, end=', sum=')
        print(i_unit.sum(), end=': ')
        print(o_unit)


if __name__ == '__main__':
    main()
