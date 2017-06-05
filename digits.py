# coding: utf-8

import numpy as np
from PIL import Image
from moyf.ml import nn, common

def main():
    digits = [[Image.open('imgs/digits/%s_%s.png' % (i, j) for j in range(10)] for i in range(10)]

    digits_vectors = [np.array([[[0 if digit.getpixel((x, y))[0] == 255 else 1 for x in range(64)] for y in range(64)] for digit in one_digits]) for one_digits in digits]
    output_vectors = [np.array([[1 if i == digit_number else 0 for i in range(10)] for digit in one_digits]) for digit_number, one_digits in enumerate(digits)]

    input_train = np.concatenate([one_digits_vectors[:9] for one_digits_vectors in digits_vectors], axis=0)
    output_train = np.concatenate([one_output_vectors[:9] for one_output_vectors in output_vectors], axis=0)
    input_result = np.concatenate([one_digits_vectors[9:10] for one_digits_vectors in digits_vectors], axis=0)

    model = nn.Model(loss=common.cross_entropy)

    model.layers.add(nn.Layer(

if __name__ == '__main__':
    main()
