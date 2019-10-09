'''
This is a nerual nets base function upon which the GAN model was built.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_impl
import numpy as np


class NN_Base(object):
    def __init__(self, batch_norm_decay=0.9,
                 batch_norm_epsilon=1e-5):
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon

    def forward_pass(self, x):
        raise NotImplementedError(
            'forward_pass() is implemented in Model sub classes')

    def _linear_fc(self, input_, output_size, scope=None, bias_start=0.0, use_bias = True,
                   kernel_initializer = tf.random_normal_initializer(0.02)):
        """
        Usually used for convert the latent vector z to the conv feature pack
        :param input_:
        :param output_size:
        :param scope:
        :param stddev:
        :param bias_start:
        :param with_w:
        :return:
        """
        with tf.variable_scope(scope):
            x = tf.layers.dense(
                inputs=input_,
                units=output_size,
                use_bias=use_bias,
                kernel_initializer= kernel_initializer,
                bias_initializer=tf.constant_initializer(bias_start),
                name=scope
            )
        return x

    def _WN_dense(self, input_, output_size, scope, init_scale = 1.0, init = False):
        """
        Weight normalization dense layer
        :param input_:
        :param output_size:
        :param scope:
        :param init_scale:
        :param init:
        :return:
        """
        with tf.variable_scope(scope):
            xs = input_.shape.as_list()
            V = tf.get_variable('V', [xs[1], output_size], tf.float32, tf.random_normal_initializer(0, 0.05))
            g = tf.get_variable('g', [output_size], dtype=tf.float32, initializer=tf.constant_initializer(1.))
            b = tf.get_variable('b', [output_size], dtype=tf.float32, initializer=tf.constant_initializer(0.))

            V_norm = tf.nn.l2_normalize(V, [0])
            x = tf.matmul(input_, V_norm)
            if init:
                mean, var = tf.nn.moments(x, [0])
                g = tf.assign(g, init_scale / tf.sqrt(var + 1e-10))
                b = tf.assign(b, -mean * g)
            x = tf.reshape(g, [1, output_size]) * x + tf.reshape(b, [1, output_size])
        return x

    def _WN_conv2d(self, input_, output_dim, k_h = 5, k_w = 5,
                   d_h = 2, d_w = 2, padding = 'SAME', init_scale = 1.0, init= False, name = "conv2d"):
        """
        Weight normalization conv2d layer
        :param input_:
        :param output_dim:
        :param k_h:
        :param k_w:
        :param d_h:
        :param d_w:
        :param padding:
        :param init_scale:
        :param init:
        :param name:
        :return:
        """
        output_dim = int(output_dim)
        strides = [1] + [d_h, d_w] + [1]

        with tf.variable_scope(name):
            xs = input_.shape.as_list()
            V = tf.get_variable('V', [k_h, k_w] + [xs[-1], output_dim],
                                tf.float32, tf.random_normal_initializer(0, 0.05))
            g = tf.get_variable('g', [output_dim], dtype=tf.float32, initializer=tf.constant_initializer(1.))
            b = tf.get_variable('b', [output_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.))

            V_norm = tf.nn.l2_normalize(V, [0, 1, 2])
            x = tf.nn.conv2d(input_, V_norm, strides, padding)
            if init:
                mean, var = tf.nn.moments(x, [0, 1, 2])
                g = tf.assign(g, init_scale / tf.sqrt(var + 1e-10))
                b = tf.assign(b, -mean * g)
            x = tf.reshape(g, [1, 1, 1, output_dim]) * x + tf.reshape(b, [1, 1, 1, output_dim])
        return x

    def _minibatch_discrimination(self, input, num_kernels, dim_per_kernel=5, name="minibatch_discrim"):
        with tf.name_scope(name):
            batch_size = input.shape[0]
            num_features = input.shape[1]
            W = tf.get_variable(name="w",
                                shape=[num_features, num_kernels * dim_per_kernel],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="b",
                                shape=[num_kernels],
                                initializer=tf.constant_initializer(0.0))
            activation = tf.matmul(input, W)
            activation = tf.reshape(activation, [batch_size, num_kernels, dim_per_kernel])
            tmp1 = tf.expand_dims(activation, 3)
            tmp2 = tf.transpose(activation, perm=[1, 2, 0])
            tmp2 = tf.expand_dims(tmp2, 0)
            abs_diff = tf.reduce_sum(tf.abs(tmp1 - tmp2), reduction_indices=[2])
            f1 = tf.reduce_sum(tf.exp(-abs_diff), reduction_indices=[2])
            f1 = f1 + b
        return f1

    def _WN_deconv2d(self, input_, output_dim, k_h = 3, k_w = 3,
                   d_h = 2, d_w = 2, padding = 'SAME', init_scale = 1.0, init= False, name = "deconv2d"):
        num_filters = int(output_dim)
        xs = input_.shape.as_list()
        if padding == 'SAME':
            target_shape = [xs[0], xs[1] * d_h,
                            xs[2] * d_w, num_filters]
        else:
            target_shape = [xs[0], xs[1] * d_h + k_h -
                            1, xs[2] * d_w + k_w - 1, num_filters]
        with tf.variable_scope(name):
            V = tf.get_variable('V',
                                [k_h, k_w] + [num_filters, xs[-1]],
                                tf.float32,
                                tf.random_normal_initializer(0, 0.05))
            g = tf.get_variable('g', [num_filters], dtype=tf.float32, initializer=tf.constant_initializer(1.))
            b = tf.get_variable('b', [num_filters], dtype=tf.float32, initializer=tf.constant_initializer(0.))

            V_norm = tf.nn.l2_normalize(V, [0, 1, 3])
            x = tf.nn.conv2d_transpose(input_, V_norm, target_shape, [1] + [d_h, d_w] + [1], padding)
            if init:
                mean, var = tf.nn.moments(x, [0, 1, 2])
                g = tf.assign(g, init_scale / tf.sqrt(var + 1e-10))
                b = tf.assign(b, -mean * g)
            x = tf.reshape(g, [1, 1, 1, num_filters]) * x + tf.reshape(b, [1, 1, 1, num_filters])
        return x

    def _conv2d(self, input_, output_dim,
                k_h=5, k_w=5, d_h=2, d_w=2,
                kernel_initializer = tf.truncated_normal_initializer(0.02), name="conv2d"):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(inputs=input_,
                                 filters=output_dim,
                                 kernel_size=(k_h, k_w),
                                 strides=(d_h, d_w),
                                 padding='same',
                                 kernel_initializer = kernel_initializer,
                                 name = name)
        return x

    def _conv_transpose(self, x, filters, kernel_size, strides, padding='SAME',
                        trainable = True):
        x = tf.layers.conv2d_transpose(inputs = x, filters = filters,
                                       kernel_size = kernel_size, strides = strides,
                                       padding = padding, trainable = trainable)
        return x

    def _relu(self, x):
        return tf.nn.relu(x)

    def _leaky_relu(self, x, alpha):
        return tf.nn.leaky_relu(x, alpha, name='leaky_relu')

    def _softplus(self, x):
        return tf.nn.softplus(x, name='softplus')

    def _fully_connected(self, x, out_dim, use_bias=True, name = None):
        x = tf.layers.dense(x, out_dim, use_bias=use_bias, name = name)
        return x

    def _drop_out(self, x, rate=0.5, train = False):
        return tf.layers.dropout(x, rate=rate, training = train)

    def _add_noise(self, inputs, mean=0.0, stddev=0.001):
        with tf.name_scope('Add_Noise'):
            noise = tf.random_normal(shape=tf.shape(inputs),
                                     mean=mean,
                                     stddev=stddev,
                                     dtype=inputs.dtype,
                                     name='noise'
                                     )
            inputs = inputs + noise
        return inputs

    def _nin(self, input, num_units, name):
        """ a network in network layer (1x1 CONV) """
        s = input.shape.as_list()
        x = tf.reshape(input, [np.prod(s[:-1]), s[-1]])
        x = self._WN_dense(x, num_units, name)
        return tf.reshape(x, s[:-1] + [num_units])

    def _conv_batch_relu(self, x, filters, kernel_size, strides):
        x = self._conv(x, filters, kernel_size, strides)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x

    def _conv_transpose_batch_relu(self, x, filters, kernel_size, strides):
        x = self._conv_transpose(x, filters, kernel_size, strides)
        x = self._batch_norm(x)
        x = self._relu(x)
        return x

    def _conv_batch_leaky_relu(self, x, filters, kernel_size, strides, alpha, padding = 'SAME'):
        x = self._conv(x, filters, kernel_size, strides, padding)
        x = self._batch_norm(x)
        x = self._leaky_relu(x, alpha)
        return x

    def _batch_norm_contrib(self, x, name, train=False):
        x = tf.contrib.layers.batch_norm(x,
                                         decay=self._batch_norm_decay,
                                         updates_collections=None,
                                         epsilon=self._batch_norm_epsilon,
                                         scale=True,
                                         is_training=train,
                                         scope=name)
        return x

    def _conv_cond_concat(self, x, y):
        """Concatenate conditioning vector on feature map axis."""
        x_shapes = x.get_shape()
        y_shapes = y.get_shape()
        return tf.concat([
            x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

    def _deconv2d(self, input_, output_shape,
                  k_h=5, k_w=5, d_h=2, d_w=2,
                  name="deconv2d", use_bias = True, kernel_initializer = tf.random_normal_initializer(0.02)):
        with tf.variable_scope(name):
            x = tf.layers.conv2d_transpose(inputs=input_,
                                           filters=output_shape,
                                           kernel_size=(k_h, k_w),
                                           strides=(d_h, d_w),
                                           padding='same',
                                           use_bias = use_bias,
                                           kernel_initializer = kernel_initializer,
                                           name=name)

        return x

    def _dense_WN(self,
                  inputs, units,
                  activation=None,
                  weight_norm=True,
                  use_bias=True,
                  kernel_initializer=None,
                  bias_initializer=init_ops.zeros_initializer(),
                  kernel_regularizer=None,
                  bias_regularizer=None,
                  activity_regularizer=None,
                  kernel_constraint=None,
                  bias_constraint=None,
                  trainable=True,
                  name=None,
                  reuse=None):
        '''
        Dense layer using weight normalizaton
        '''
        layer = Dense(units,
                      activation=activation,
                      weight_norm=weight_norm,
                      use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer,
                      kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint,
                      trainable=trainable,
                      name=name,
                      dtype=inputs.dtype.base_dtype,
                      _scope=name,
                      _reuse=reuse)
        return layer.apply(inputs)


class Dense(core_layers.Dense):
    '''
    Dense layer implementation using weight normalization.
    Code borrowed from:
    https://github.com/llan-ml/weightnorm/blob/master/dense.py
    '''

    def __init__(self, *args, **kwargs):
        self.weight_norm = kwargs.pop("weight_norm")
        super(Dense, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(
            min_ndim=2, axes={-1: input_shape[-1].value})
        kernel = self.add_variable(
            'kernel',
            shape=[input_shape[-1].value, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.weight_norm:
            self.g = self.add_variable(
                "wn/g",
                shape=(self.units,),
                initializer=init_ops.ones_initializer(),
                dtype=kernel.dtype,
                trainable=True)
            self.kernel = nn_impl.l2_normalize(kernel, dim=0) * self.g
        else:
            self.kernel = kernel
        if self.use_bias:
            self.bias = self.add_variable(
                'bias',
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True