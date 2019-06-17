# -*- coding: utf-8 -*-
"""
Time    : 4/19/19 10:37 AM
Author  : HyperAndy
Email   : hitwangzijian@163.com
File    : alibaba_attack_ensemble.py
Software: PyCharm Community Edition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
# from tensorflow.contrib.slim.nets import inception
import scipy
from scipy.misc import imread
from scipy.misc import imresize
# from cleverhans.attacks import FastGradientMethod
# from cleverhans.attacks import Model
from PIL import Image
import pandas as pd
from tensorflow.contrib.slim.nets import inception, resnet_v1, vgg
import numpy
import sys
import PIL.Image
import PIL.ImageFile
import PIL.ImageFilter

# import inception_v3 as inception

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')
tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')
tf.flags.DEFINE_integer(
    'image_width', 224, 'Width of each input images.')
tf.flags.DEFINE_integer(
    'image_height', 224, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'batch_size', 11, 'How many images process at one time.')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'Number of Classes')
FLAGS = tf.flags.FLAGS

# Checkpoint_path = '../../models'
Checkpoint_path = './models/'
model_checkpoint_map = {
    'inception_v1': os.path.join(Checkpoint_path, 'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(Checkpoint_path, 'resnet_v1_50', 'model.ckpt-49800'),
    'vgg_16': os.path.join(Checkpoint_path, 'vgg_16', 'vgg_16.ckpt')}

(PIL.ImageFile).LOAD_TRUNCATED_IMAGES = True
NUMBER_OF_CLASSES = 110
SIZE = 299
OutputDirectory = FLAGS.output_dir
InputDirectory = FLAGS.input_dir


def preprocessor(im, model_name):
    """
    :param im:  the raw image array [height, width, channels]
    :param model_name: the network model name
    :return:
    """
    if 'inception' in model_name.lower():
        image = tf.image.resize_bilinear(im, [FLAGS.image_height, FLAGS.image_width], align_corners=False)
        # image = imresize(im, [FLAGS.image_height, FLAGS.image_width]).astype(np.float)
        image = (image / 255.0) * 2.0 - 1.0
        return image
    if 'resnet' in model_name.lower() or 'vgg' in model_name.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        image = tf.image.resize_bilinear(im, [FLAGS.image_height, FLAGS.image_width], align_corners=False)
        # image = imresize(im, [FLAGS.image_height, FLAGS.image_width]).astype(np.float)
        # image[:, :, 0] = image[:, :, 0] - _R_MEAN
        # image[:, :, 1] = image[:, :, 1] - _G_MEAN
        # image[:, :, 2] = image[:, :, 2] - _B_MEAN
        tmp_0 = image[:, :, :, 0] - _R_MEAN
        tmp_1 = image[:, :, :, 1] - _G_MEAN
        tmp_2 = image[:, :, :, 2] - _B_MEAN
        image = tf.stack([tmp_0, tmp_1, tmp_2], 3)
        return image


def main(_):
    """Run the sample attack"""
    with tf.Graph().as_default():
        # image = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
        img = tf.placeholder(tf.float32, [1, 299, 299, 3])
        # label = tf.placeholder(tf.int32, [batch_size, 110])
        # label = tf.placeholder(tf.int32, shape=[1, 1])
        y = tf.placeholder(tf.int32, shape=[1])

        processed_imgs_vgg_16 = preprocessor(img, 'vgg')
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
                processed_imgs_vgg_16, num_classes=110, is_training=False, scope='vgg_16', spatial_squeeze=False)

        conv_layer = end_points_vgg_16['vgg_16/pool5']  # while/vgg_16/pool5
        end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
        end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])

        one_hot = tf.one_hot(y, 110)
        # one_hot = tf.sparse_to_dense(label, [110], 1.0)
        # cost = (-1) * tf.reduce_sum(tf.multiply(one_hot, tf.log(end_points_vgg_16['probs'])), axis=1)
        signal = tf.multiply(end_points_vgg_16['logits'], one_hot)
        loss = tf.reduce_mean(signal)
        grads = tf.gradients(loss, conv_layer)[0]
        # Normalizing the gradients
        norm_grads = tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

        s = tf.train.Saver()
        # Prepare graph
        with tf.Session() as sess:
            File = open(InputDirectory + "/dev.csv")
            File.readline()

            # s.restore(sess, '../vgg_16/vgg_16.ckpt')
            s.restore(sess, model_checkpoint_map['vgg_16'])

            for line in File:
                split_line = line.strip().split(",")

                filename = split_line[0]
                true_label = int(split_line[1])
                # true_label = np.array([true_label]).astype(np.int32)
                targeted_label = int(split_line[2])

                # image_pil = PIL.Image.open(OutputDirectory + "/" + filename)
                # image_pil_origin = PIL.Image.open(InputDirectory + "/" + filename)
                # image = numpy.asarray(image_pil.resize([SIZE, SIZE], PIL.Image.BILINEAR).convert("RGB")).astype(
                #     numpy.float32)
                # image_origin = numpy.asanyarray(
                #     image_pil_origin.resize([SIZE, SIZE], PIL.Image.BILINEAR).convert("RGB")).astype(numpy.float32)
                image = imread(OutputDirectory + "/" + filename, mode='RGB')
                image_origin = imread(InputDirectory + "/" + filename, mode='RGB')
                img_origin = np.expand_dims(image_origin, axis=0)

                output, grads_val, one_hot1 = sess.run([conv_layer, norm_grads, one_hot], feed_dict={img: img_origin, y: [true_label]})
                # cam3, cam = grad_cam([image_origin], true_label, sess)
                output = output[0]  # [7,7,512]
                grads_val = grads_val[0]  # [7,7,512]

                weights = np.mean(grads_val, axis=(0, 1))  # [512]
                cam = np.ones(output.shape[0: 2], dtype=np.float32)  # [7,7]

                # Taking a weighted average
                for i, w in enumerate(weights):
                    cam += w * output[:, :, i]

                # Passing through ReLU
                cam = np.maximum(cam, 0)
                cam = cam / np.max(cam)
                cam = imresize(cam, (299, 299))

                # Converting grayscale to 3-D
                cam3 = np.expand_dims(cam, axis=2)
                cam3 = np.tile(cam3, [1, 1, 3])
                new_image = np.where(cam3 > np.mean(cam3) / 1.1, image, image_origin)

                PIL.Image.fromarray(numpy.asarray(new_image, numpy.int8), "RGB").save(
                    OutputDirectory + "/" + filename)

            File.close()


if __name__ == '__main__':
    tf.app.run()
    # pass
