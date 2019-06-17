# coding: utf-8
"""
https://github.com/Ankush96/grad-cam.tensorflow/blob/master/main.py
"""
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import numpy as np
import os
import sys
from tensorflow.contrib.slim.nets import inception, resnet_v1, vgg
from scipy.misc import imread, imresize
from skimage import io

# from matplotlib import pyplot as plt

slim = tf.contrib.slim

# @ops.RegisterGradient("GuideRelu")
# def _GuideReluGrad(op, grad):
#     return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

batch_size = 1


# eval_graph = tf.Graph()
# with eval_graph.as_default():
#     with eval_graph.gradient_override_map({'relu':'GuideRelu'}):

def load_image(img_path):
    print("Loading image")
    img = imread(img_path, mode='RGB')
    img = imresize(img, (224, 224)).astype(np.float)
    # Converting shape from [224,224,3] tp [1,224,224,3]
    x = np.expand_dims(img, axis=0)
    # Converting RGB to BGR for VGG
    # x = x[:, :, :, ::-1]
    return x, img


def preprocess_for_model(images):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    # images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
    tmp_0 = images[:, :, :, 0] - _R_MEAN
    tmp_1 = images[:, :, :, 1] - _G_MEAN
    tmp_2 = images[:, :, :, 2] - _B_MEAN
    images = tf.stack([tmp_0, tmp_1, tmp_2], 3)
    return images


def grad_cam(x, y, sess):
    image = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    # label = tf.placeholder(tf.int32, [batch_size, 110])
    # label = tf.placeholder(tf.int32, shape=[batch_size])

    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
            preprocess_for_model(image), num_classes=110, is_training=False, scope='vgg_16', spatial_squeeze=False)

    conv_layer = end_points_vgg_16['vgg_16/pool5']  # while/vgg_16/pool5
    end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
    end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])

    one_hot = tf.one_hot(y, 110)
    # one_hot = tf.sparse_to_dense(y, [110], 1.0)
    # cost = (-1) * tf.reduce_sum(tf.multiply(one_hot, tf.log(end_points_vgg_16['probs'])), axis=1)
    signal = tf.multiply(end_points_vgg_16['logits'], one_hot)
    loss = tf.reduce_mean(signal)
    grads = tf.gradients(loss, conv_layer)[0]
    # Normalizing the gradients
    norm_grads = tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

    s = tf.train.Saver()
    s.restore(sess, '../vgg_16/vgg_16.ckpt')
    output, grads_val, one_hot1 = sess.run([conv_layer, norm_grads, one_hot], feed_dict={image: x})  # , label: y

    # print one_hot1
    output = output[0]  # [7,7,512]
    # print output.shape
    grads_val = grads_val[0]  # [7,7,512]
    # print grads_val.shape

    weights = np.mean(grads_val, axis=(0, 1))  # [512]
    # print weights.shape
    cam = np.ones(output.shape[0: 2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # print cam

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    # print cam
    cam = cam / np.max(cam)
    # print cam
    # cam = np.resize(cam, (224, 224))
    # print cam
    cam = imresize(cam, (224, 224))

    # Converting grayscale to 3-D
    cam3 = np.expand_dims(cam, axis=2)
    cam3 = np.tile(cam3, [1, 1, 3])
    # print cam3

    # gb_grad = tf.gradients(cost, image)[0]
    # init = tf.global_variables_initializer()
    return cam3, cam


def main(_):
    x, img = load_image('../dev_data/0bc58747-9a2e-43e1-af1f-cf0a41f9f2ba.png')
    y = 39
    with tf.Graph().as_default():
        # s = tf.train.Saver()
        with tf.Session() as sess:
            # s.restore(sess, '../../models/vgg_16/vgg_16.ckpt')
            # sess = tf.Session()
            # s = tf.train.Saver()
            # s.restore(sess, '../../models/vgg_16/vgg_16.ckpt')
            cam3, cam = grad_cam([img], y, sess)
            cam3 = np.where(cam3 > np.mean(cam3)/2, 1.0, 0.0)

            # cam3 /= cam3.max()
            print cam3

            img = img.astype(float)
            img /= img.max()

            new_image = np.multiply(img, cam3)
            print new_image
            # new_image /= new_image.max()
            # print(new_image)
            # io.imshow(new_image)
            # plt.show()
            # cam = imresize(cam, (224, 224))
            # io.imsave('./cam1.png', cam)
            # io.imsave('./cam.png', cam3)
            io.imsave('./test.png', new_image)
            io.imsave('./origin.png', img)


if __name__ == '__main__':
    tf.app.run()
    # main()
