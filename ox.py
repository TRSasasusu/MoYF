# coding: utf-8

import numpy as np
from PIL import Image
from moyf.ml import nn, common

def main():
    o_imgs = [Image.open('imgs/ox/o%s.png' % i, 'r') for i in range(19)]
    x_imgs = [Image.open('imgs/ox/x%s.png' % i, 'r') for i in range(19)]

    o_img_vectors = []
    x_img_vectors = []
    for o_img, x_img in zip(o_imgs, x_imgs):
        o_img_vectors.append([0 if o_img.getpixel((x, y))[0] == 255 else 1 for x in range(64) for y in range(64)])
        x_img_vectors.append([0 if x_img.getpixel((x, y))[0] == 255 else 1 for x in range(64) for y in range(64)])
    o_img_vectors = np.array(o_img_vectors)
    x_img_vectors = np.array(x_img_vectors)

    input_train = np.append(o_img_vectors[:17], x_img_vectors[:17], axis=0)
    output_train = np.array([[1, 0] * 17, [0, 1] * 17]).reshape(34, 2)
    input_result = np.append(o_img_vectors[17:19], x_img_vectors[17:19], axis=0)

    model = nn.Model(loss=common.cross_entropy)

    model.layers.add(nn.Layer(input_num=64*64, output_num=800, activate=common.relu))
    model.layers.add(nn.Layer(output_num=200, activate=common.relu))
    model.layers.add(nn.Layer(output_num=2, activate=common.softmax))

    model.learn(input_train=input_train, output_train=output_train, epoch_num=1000)

    output_result = model.result(input_result=input_result)

    for i in range(2):
        print('o%s.png: %s' % (i + 17, output_result[i][0]))
        print('x%s.png: %s' % (i + 17, output_result[i + 2][0]))

if __name__ == '__main__':
    main()
