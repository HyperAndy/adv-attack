"""
Implementation of example defense.
This defense loads inception v1 checkpoint and classifies all images using loaded checkpoint.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
#from tensorflow.contrib.slim.nets import inception
from scipy.misc import imread
from scipy.misc import imresize
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import Model
from PIL import Image
import pandas as pd
# from tensorflow.contrib.slim.nets import inception
import inception_v1 as inception

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
    'image_resize', 200, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'num_classes', 110, 'Number of Classes')
tf.flags.DEFINE_float(
    'momentum', 1.0, 'Momentum.')
tf.flags.DEFINE_integer(
    'num_iter', 20, 'Number of iterations.')
tf.flags.DEFINE_float(
    'prob', 0.5, 'probability of using diverse inputs.')
tf.flags.DEFINE_float(
    'max_epsilon', 32.0, 'Maximum size of adversarial perturbation.')
FLAGS = tf.flags.FLAGS

CHECKPOINTS_DIR = './models/'
model_checkpoint_map = {
    'inception_v1': os.path.join(CHECKPOINTS_DIR, 'inception_v1', 'inception_v1.ckpt'),
    'resnet_v1_50': os.path.join(CHECKPOINTS_DIR, 'resnet_v1_50', 'model.ckpt-49800'),
    'vgg_16': os.path.join(CHECKPOINTS_DIR, 'vgg_16', 'vgg_16.ckpt')}


def load_images(input_dir):
    images = []
    filenames = []
    true_labels = []
    idx = 0
    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2label = {dev.iloc[i]['filename']: dev.iloc[i]['trueLabel'] for i in range(len(dev))}
    for filename in filename2label.keys():
        raw_image = imread(os.path.join(input_dir, filename), mode='RGB')
        # image = tf.image.resize_bilinear(raw_image, [FLAGS.image_height,FLAGS.image_width],align_corners=False)
        image = imresize(raw_image, [FLAGS.image_height, FLAGS.image_width]).astype(np.float)
        image = (image / 255.0) * 2.0 - 1.0
        images.append(image)
        filenames.append(filename)
        true_labels.append(filename2label[filename])
        idx += 1
        if idx == FLAGS.batch_size:
            images = np.array(images)
            yield filenames, images, true_labels
            filenames = []
            images = []
            true_labels = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        yield filenames, images, true_labels


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            # resize back to [299, 299]
            r_img = imresize(img, [299, 299])
            Image.fromarray(r_img).save(f, format='PNG')


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def input_diversity(input_tensor):
    print('begin')
    rnd = tf.random_uniform((), FLAGS.image_resize,  FLAGS.image_width, dtype=tf.int32)
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
    print(padded)
    print(input_tensor)
    # return tf.cond(tf.less(tf.random_uniform(shape=[1])[0], tf.constant(FLAGS.prob)), lambda: padded, lambda: input_tensor)
    return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)


def graph(x, y, i, x_max, x_min, grad):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum
    num_classes = 110

    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits_inc_v1, end_point_inc_v1 = inception.inception_v1(
            # input_diversity(x), num_classes=num_classes, is_training=False, scope='InceptionV1')
            x, num_classes=num_classes, is_training=False, scope='InceptionV1')
    pred = tf.argmax(end_point_inc_v1['Predictions'], 1)
    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred + (1 - first_round) * y
    one_hot = tf.one_hot(y, num_classes)
    logits = end_point_inc_v1['Logits']
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot,
                                                    logits,
                                                    label_smoothing=0.0,
                                                    weights=1.0)
    noise = tf.gradients(cross_entropy, x)[0]
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    noise = momentum * grad + noise
    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise


def stop(x, y, i, x_max, x_min, grad):
    num_iter = FLAGS.num_iter
    # num_iter = int(min(FLAGS.max_epsilon + 4, 1.25 * FLAGS.max_epsilon))
    return tf.less(i, num_iter)


def main(_):
    """Run the sample attack"""
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    nb_classes = FLAGS.num_classes
    # tf.logging.set_verbosity(tf.logging.INFO)
    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        print(x_input)
        print(input_diversity(x_input))
        x_adv, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad])

        # Run computation
        #s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV1'))
        var = tf.global_variables()
        var_to_restore = [val for val in var]
        s1 = tf.train.Saver(var_to_restore)

        with tf.Session() as sess:
            s1.restore(sess, FLAGS.checkpoint_path)

            for filenames, images, true_labels in load_images(FLAGS.input_dir):
                # print('attack begin')
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)
                # print('attack end')


if __name__ == '__main__':
    # FLAGS.input_dir = '/Users/wzj/Downloads/alibaba/dev_data/'
    # FLAGS.output_dir = '/Users/wzj/Downloads/alibaba/out1/'
    # FLAGS.checkpoint_path = '../inception_v1/inception_v1.ckpt'
    tf.app.run()
    # pass
