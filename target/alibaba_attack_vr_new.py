# -*- coding: utf-8 -*-
"""
Time    : 5/28/19 3:47 PM
Author  : HyperAndy
Email   : hitwangzijian@163.com
File    : alibaba_attack_vr_new.py
Software: PyCharm Community Edition
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1, inception, vgg
from scipy.misc import imread
from scipy.misc import imresize
import scipy
# from cleverhans.attacks import FastGradientMethod
# from cleverhans.attacks import Model
from PIL import Image
import pandas as pd
import numpy
import PIL.Image
import PIL.ImageFile
import PIL.ImageFilter

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')
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
tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')
tf.flags.DEFINE_integer(
    'num_iter', 10, 'Number of iterations.')
tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
tf.flags.DEFINE_integer(
    'image_resize', 170, 'Height of each input images.')
tf.flags.DEFINE_float(
    'prob', 0.5, 'probability of using diverse inputs.')
FLAGS = tf.flags.FLAGS

Checkpoint_path = './models/'
# Checkpoint_path = '../'
# Checkpoint_path = '../../models/'
model_checkpoint_map = {
    'inception_v1': os.path.join(Checkpoint_path, 'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(Checkpoint_path, 'resnet_v1_50', 'model.ckpt-49800'),
    'vgg_16': os.path.join(Checkpoint_path, 'vgg_16', 'vgg_16.ckpt')}

(PIL.ImageFile).LOAD_TRUNCATED_IMAGES = True
NUMBER_OF_CLASSES = 110
SIZE = 299
OutputDirectory = FLAGS.output_dir
InputDirectory = FLAGS.input_dir


def crop(rand_begin, rand_end):
    File = open(InputDirectory + "/dev.csv")
    File.readline()

    for line in File:
        split_line = line.strip().split(",")

        filename = split_line[0]
        true_label = int(split_line[1])
        targeted_label = int(split_line[2])

        image_pil = PIL.Image.open(OutputDirectory + "/" + filename)
        image_pil_origin = PIL.Image.open(InputDirectory + "/" + filename)
        image = numpy.asarray(image_pil.resize([SIZE, SIZE], PIL.Image.BILINEAR).convert("RGB")).astype(numpy.float32)
        image_origin = numpy.asanyarray(
            image_pil_origin.resize([SIZE, SIZE], PIL.Image.BILINEAR).convert("RGB")).astype(numpy.float32)

        rnd = np.random.randint(rand_begin, rand_end)
        rnd1 = 299 - rnd

        new_image = image_origin.copy()
        new_image[rnd:rnd1, rnd:rnd1] = image[rnd:rnd1, rnd:rnd1]

        PIL.Image.fromarray(numpy.asarray(new_image, numpy.int8), "RGB").save(OutputDirectory + "/" + filename)

    File.close()


def preprocess(images, model_type):
    if 'inception' in model_type.lower():
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        # images = imresize(images, [FLAGS.image_height, FLAGS.image_width]).astype(np.float)
        # tensor-scalar operation
        images = (images / 255.0) * 2.0 - 1.0
        return images

    if 'resnet' in model_type.lower() or 'vgg' in model_type.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        # images = imresize(images, [FLAGS.image_height, FLAGS.image_width]).astype(np.float)
        tmp_0 = images[:, :, :, 0] - _R_MEAN
        tmp_1 = images[:, :, :, 1] - _G_MEAN
        tmp_2 = images[:, :, :, 2] - _B_MEAN
        images = tf.stack([tmp_0, tmp_1, tmp_2], 3)
        return images


def load_images(input_dir):
    images = []
    filenames = []
    target_labels = []
    idx = 0
    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2label = {dev.iloc[i]['filename']: dev.iloc[i]['targetedLabel'] for i in range(len(dev))}
    for filename in filename2label.keys():
        image = imread(os.path.join(input_dir, filename), mode='RGB')
        # image = tf.image.resize_bilinear(raw_image, [FLAGS.image_height,FLAGS.image_width], align_corners=False)
        # image = imresize(raw_image, [FLAGS.image_height, FLAGS.image_width]).astype(np.float)
        # image = (image / 255.0) * 2.0 - 1.0
        images.append(image)
        filenames.append(filename)
        target_labels.append(filename2label[filename])
        idx += 1
        if idx == FLAGS.batch_size:
            images = np.array(images)
            yield filenames, images, target_labels
            filenames = []
            images = []
            target_labels = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        yield filenames, images, target_labels


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        image = (((images[i] + 1.0) * 0.5) * 255.0).astype(np.uint8)
        image = imresize(image, [299, 299])
        Image.fromarray(image).save(os.path.join(output_dir, filename), format='PNG')


def input_diversity(input_tensor):
    # print('begin')
    rnd = tf.random_uniform((), FLAGS.image_resize, FLAGS.image_width, dtype=tf.int32)
    rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # rescaled = imresize(input_tensor, [rnd, rnd]).astype(np.float)
    h_rem = FLAGS.image_width - rnd
    w_rem = FLAGS.image_width - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_width, FLAGS.image_width, 3))
    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def graph_1(x, x_min, x_max, noise_vr, m, noise_pre):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    alpha = eps / FLAGS.num_iter
    # num_classes = 110
    # m = 20
    # sigma = 5.0

    noise = noise_vr / m
    noise = noise / tf.reshape(
        tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1),
        [FLAGS.batch_size, 1, 1, 1])
    noise = FLAGS.momentum * noise_pre + noise
    noise = noise / tf.reshape(
        tf.contrib.keras.backend.std(tf.reshape(noise, [FLAGS.batch_size, -1]), axis=1),
        [FLAGS.batch_size, 1, 1, 1])
    x = x - alpha * tf.clip_by_value(tf.round(noise), -2, 2)
    x = tf.clip_by_value(x, x_min, x_max)
    return x, noise


def graph_2(x, y, noise_vr):
    num_classes = 110
    # batch_shape = [FLAGS.batch_size, 224, 224, 3]
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits_inc_v1, end_points_inc_v1 = inception.inception_v1(
            x, num_classes=num_classes, is_training=False, scope='InceptionV1')

    # rescale pixle range from [-1, 1] to [0, 255] for resnet_v1 and vgg's input
    image = (((x + 1.0) * 0.5) * 255.0)
    processed_imgs_res_v1_50 = preprocess(image, 'resnet_v1_50')
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits_res_v1_50, end_points_res_v1_50 = resnet_v1.resnet_v1_50(
            processed_imgs_res_v1_50, num_classes=num_classes, is_training=False, scope='resnet_v1_50')

    end_points_res_v1_50['logits'] = tf.squeeze(end_points_res_v1_50['resnet_v1_50/logits'], [1, 2])
    end_points_res_v1_50['probs'] = tf.nn.softmax(end_points_res_v1_50['logits'])

    processed_imgs_vgg_16 = preprocess(image, 'vgg_16')
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits_vgg_16, end_points_vgg_16 = vgg.vgg_16(
            processed_imgs_vgg_16, num_classes=num_classes, is_training=False, scope='vgg_16')

    end_points_vgg_16['logits'] = end_points_vgg_16['vgg_16/fc8']
    end_points_vgg_16['probs'] = tf.nn.softmax(end_points_vgg_16['logits'])

    one_hot = tf.one_hot(y, num_classes)
    ########################
    logits = (end_points_inc_v1['Logits'] + end_points_res_v1_50['logits'] + end_points_vgg_16['logits']) / 5.0
    # logits = (end_points_inc_v1['Logits'] + end_points_res_v1_50['logits']) / 2.0
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    noise_vr = tf.add(noise_vr, tf.gradients(cross_entropy, x)[0])
    return noise_vr


# Momentum Iterative FGSM
def main(_):
    # some parameter
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, 224, 224, 3]
    m = 5
    sigma = 2.0

    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        raw_inputs = tf.placeholder(tf.uint8, shape=[None, 299, 299, 3])
        processed_imgs = preprocess(raw_inputs, 'inception_v1')

        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
        m1 = tf.constant(m, tf.float32)

        #     y = tf.constant(np.zeros([batch_size]), tf.int64)
        y = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        noise_vr1 = tf.placeholder(tf.float32, shape=batch_shape)
        noise_pre1 = tf.placeholder(tf.float32, shape=batch_shape)
        # i = tf.constant(0)
        # grad = tf.zeros(shape=batch_shape)
        # x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad])
        noise1 = graph_2(x_input, y, noise_vr1)
        adv, noise2 = graph_1(x_input, x_min, x_max, noise_vr1, m1, noise_pre1)


        # Run computation
        # print(slim.get_model_variables(scope='InceptionV1'))
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='resnet_v1_50'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='vgg_16'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v1'])
            s2.restore(sess, model_checkpoint_map['resnet_v1_50'])
            s3.restore(sess, model_checkpoint_map['vgg_16'])

            # graph1 = graph_1()
            # graph2 = graph_2()
            for filenames, images, target_labels in load_images(FLAGS.input_dir):
                # print('attack begin')
                processed_image = sess.run(processed_imgs, feed_dict={raw_inputs: images})
                noise_pre = np.zeros(batch_shape, dtype=np.float32)
                adv_images = processed_image
                for i in range(FLAGS.num_iter):
                    # print(i)
                    noise_vr = np.zeros(batch_shape, dtype=np.float32)
                    for j in range(m):
                        # print("go")
                        processed_image1 = sess.run(
                            tf.random_normal(processed_image.shape, mean=processed_image, stddev=sigma))
                        noise_vr = sess.run(noise1, feed_dict={x_input:processed_image1, y:target_labels, noise_vr1:noise_vr})
                        # print(noise_vr)
                    # print("gogo")
                    adv_images, noise_pre = sess.run([adv, noise2], feed_dict={x_input:adv_images, noise_vr1:noise_vr, noise_pre1:noise_pre})

                save_images(adv_images, filenames, FLAGS.output_dir)
                # print('attack end')

                # crop(10, 15)


if __name__ == '__main__':
    tf.app.run()
