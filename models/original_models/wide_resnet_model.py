from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from cleverhans.serial import Model


class Layer(object):
    def get_output_shape(self):
        return self.output_shape


class WideResnetModel(Model):
    """ResNet model."""

    def __init__(self, image_height=32, image_width=32, n_channels=3, n_classes=10):
        """ResNet constructor.

        :param layers: a list of layers in CleverHans format
          each with set_input_shape() and fprop() methods.
        :param input_shape: 4-tuple describing input shape (e.g None, 32, 32, 3)
        """
        super().__init__(scope='', nb_classes=n_classes, hparams={})
        self._image_height = image_height
        self._image_width = image_width
        self._n_channels = n_channels
        self._n_classes = n_classes
        self._layers = [Input(), Conv2D(), Flatten(), Linear(self._n_classes), Softmax()]
        self._layer_names = None
        self._input_shape = (None, self._image_height, self._image_width, self._n_channels)
        self._build()

    def get_vars(self):
        if hasattr(self, "vars"):
            return self.vars
        return super(WideResnetModel, self).get_vars()

    def _build(self):
        self._layer_names = []
        input_shape = self._input_shape
        if isinstance(self._layers[-1], Softmax):
            self._layers[-1].name = 'probs'
            self._layers[-2].name = 'logits'
        else:
            self._layers[-1].name = 'logits'
        for i, layer in enumerate(self._layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
                layer.name = name
            self._layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def make_input_placeholder(self):
        return tf.placeholder(tf.float32, self._input_shape)

    def make_label_placeholder(self):
        return tf.placeholder(tf.float32, (None, self._n_classes))

    def fprop(self, x, set_ref=False):
        states = []
        for layer in self._layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self._layer_names, states))
        return states

    def add_internal_summaries(self):
        pass


def _stride_arr(stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]


class Input(Layer):
    def __init__(self):
        self.output_shape = None

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        # assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        with tf.compat.v1.variable_scope('input', reuse=tf.compat.v1.AUTO_REUSE):
            input_standardized = tf.map_fn(
                lambda img: tf.image.per_image_standardization(img), x)
            return _conv('init_conv', input_standardized,
                         3, 3, 16, _stride_arr(1))


class Conv2D(Layer):
    def __init__(self):
        self.output_shape = None

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape

        # Uncomment the following codes to use w28-10 wide residual network.
        # It is more memory efficient than very deep residual network and has
        # comparably good performance.
        # https://arxiv.org/pdf/1605.07146v1.pdf
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        # Update hps.num_residual_units to 9
        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        filters = [16, 160, 320, 640]
        res_func = _residual
        with tf.compat.v1.variable_scope('unit_1_0', reuse=tf.compat.v1.AUTO_REUSE):
            x = res_func(x, filters[0], filters[1], _stride_arr(strides[0]),
                         activate_before_residual[0])
        for i in range(1, 5):
            with tf.compat.v1.variable_scope(('unit_1_%d' % i), reuse=tf.compat.v1.AUTO_REUSE):
                x = res_func(x, filters[1], filters[1],
                             _stride_arr(1), False)

        with tf.compat.v1.variable_scope('unit_2_0', reuse=tf.compat.v1.AUTO_REUSE):
            x = res_func(x, filters[1], filters[2], _stride_arr(strides[1]),
                         activate_before_residual[1])
        for i in range(1, 5):
            with tf.compat.v1.variable_scope(('unit_2_%d' % i), reuse=tf.compat.v1.AUTO_REUSE):
                x = res_func(x, filters[2], filters[2],
                             _stride_arr(1), False)

        with tf.compat.v1.variable_scope('unit_3_0', reuse=tf.compat.v1.AUTO_REUSE):
            x = res_func(x, filters[2], filters[3], _stride_arr(strides[2]),
                         activate_before_residual[2])
        for i in range(1, 5):
            with tf.compat.v1.variable_scope(('unit_3_%d' % i), reuse=tf.compat.v1.AUTO_REUSE):
                x = res_func(x, filters[3], filters[3],
                             _stride_arr(1), False)

        with tf.compat.v1.variable_scope('unit_last', reuse=tf.compat.v1.AUTO_REUSE):
            x = _batch_norm('final_bn', x)
            x = _relu(x, 0.1)
            x = _global_avg_pool(x)

        return x


class Linear(Layer):
    def __init__(self, num_hid):
        self.num_hid = num_hid
        self.input_shape = None
        self.dim = None
        self.output_shape = None

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.dim = dim
        self.output_shape = [batch_size, self.num_hid]
        self.make_vars()

    def make_vars(self):
        with tf.compat.v1.variable_scope('logit', reuse=tf.compat.v1.AUTO_REUSE):
            w = tf.compat.v1.get_variable(
                'DW', [self.dim, self.num_hid],
                initializer=tf.initializers.variance_scaling(
                    distribution='uniform'))
            b = tf.compat.v1.get_variable('biases', [self.num_hid],
                                initializer=tf.initializers.constant())
        return w, b

    def fprop(self, x):
        w, b = self.make_vars()
        return tf.compat.v1.nn.xw_plus_b(x, w, b)


def _batch_norm(name, x):
    """Batch normalization."""
    with tf.name_scope(name):
        return tf.contrib.layers.batch_norm(
            inputs=x,
            decay=.9,
            center=True,
            scale=True,
            activation_fn=None,
            updates_collections=None,
            is_training=False)


def _residual(x, in_filter, out_filter, stride,
              activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
        with tf.compat.v1.variable_scope('shared_activation'):
            x = _batch_norm('init_bn', x)
            x = _relu(x, 0.1)
            orig_x = x
    else:
        with tf.compat.v1.variable_scope('residual_only_activation'):
            orig_x = x
            x = _batch_norm('init_bn', x)
            x = _relu(x, 0.1)

    with tf.compat.v1.variable_scope('sub1'):
        x = _conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.compat.v1.variable_scope('sub2'):
        x = _batch_norm('bn2', x)
        x = _relu(x, 0.1)
        x = _conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.compat.v1.variable_scope('sub_add'):
        if in_filter != out_filter:
            orig_x = tf.nn.avg_pool2d(orig_x, stride, stride, 'VALID')
            orig_x = tf.pad(
                orig_x, [[0, 0], [0, 0],
                         [0, 0], [(out_filter - in_filter) // 2,
                                  (out_filter - in_filter) // 2]])
        x += orig_x

    tf.compat.v1.logging.debug('image after unit %s', x.get_shape())
    return x


def _decay():
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find('DW') > 0:
            costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)


def _conv(name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        n = filter_size * filter_size * out_filters
        kernel = tf.compat.v1.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(
                stddev=np.sqrt(2.0 / n)))
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')


def _relu(x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def _global_avg_pool(x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


class Softmax(Layer):
    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x)


class Flatten(Layer):
    def __init__(self):
        self.input_shape = None
        self.output_width = None
        self.output_shape = None

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [None, output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])
