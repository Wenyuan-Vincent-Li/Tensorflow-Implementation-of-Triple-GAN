import sys, os
if os.getcwd().endswith("Path_Semi_GAN"):
    root_dir = os.getcwd()
else:
    root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
he_init = variance_scaling_initializer()
from Model import model_base
from Model import nn
import numpy as np

class Good_GAN_cifar10(model_base.NN_Base):
    def __init__(self, config):
        super(Good_GAN_cifar10, self).__init__(config.BATCH_NORM_DECAY, config.BATCH_NORM_EPSILON)
        self.config = config

    def leakyReLu(self, x, alpha=0.2, name=None):
        if name:
            with tf.variable_scope(name):
                return self._leakyReLu_impl(x, alpha)
        else:
            return self._leakyReLu_impl(x, alpha)

    def _leakyReLu_impl(self, x, alpha):
        return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))

    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    def good_generator(self, z, y, init=False, reuse=False):
        with tf.variable_scope('good_generator', reuse=reuse):
            counter = {}
            # project 'z' and reshape
            yb = tf.reshape(y, [tf.shape(y)[0], 1, 1, self.config.NUM_CLASSES])
            z = tf.concat([z, y], 1)

            h0 = self._linear_fc(z, 4 * 4 * 512, 'gg_h0_lin', kernel_initializer = he_init)
            h0 = tf.nn.relu(h0, 'gg_rl0')  # [4,4]
            h0 = self._batch_norm_contrib(h0, 'gg_bn0', train=True)
            h0 = tf.reshape(h0, [-1, 4, 4, 512])
            h0 = self._conv_cond_concat(h0, yb)

            h0 = self._deconv2d(h0, 256, k_w=5, k_h=5, d_w=2, d_h=2, name='gg_dconv0', kernel_initializer = he_init)  # [8, 8]
            h0 = tf.nn.relu(h0, 'gg_rl1')
            h0 = self._batch_norm_contrib(h0, 'gg_bn1', train=True)
            h0 = self._conv_cond_concat(h0, yb)

            h1 = self._deconv2d(h0, 128, k_w=5, k_h=5, d_w=2, d_h=2, name='gg_dconv1', kernel_initializer = he_init)
            h1 = tf.nn.relu(h1, 'gg_rl2')  # [16,16]
            h1 = self._batch_norm_contrib(h1, 'gg_bn2', train=True)
            h1 = self._conv_cond_concat(h1, yb)

            h2 = self._deconv2d(h1, 3, k_w=5, k_h=5, d_w=2, d_h=2, name = 'gg_dconv2', kernel_initializer = he_init)
            h2 = tf.nn.tanh(h2)
        return h2

    def discriminator(self, image, y, init=False, reuse = False, getter=None):
        with tf.variable_scope('discriminator', reuse=reuse, custom_getter=getter):
            counter = {}
            image = self._drop_out(image, 0.2, True)
            yb = tf.reshape(y, [tf.shape(image)[0], 1, 1, self.config.NUM_CLASSES])
            image = self._conv_cond_concat(image, yb)

            h0 = self._conv2d(image, 32, k_h=3, k_w=3, d_h=1, d_w=1, kernel_initializer = he_init, name="conv2d_00")
            h0 = self.leakyReLu(h0)
            h0 = self._conv_cond_concat(h0, yb)

            h0 = self._conv2d(h0, 32, k_h=3, k_w=3, d_h=2, d_w=2, kernel_initializer = he_init, name="conv2d_01")
            h0 = self.leakyReLu(h0)
            h0 = self._drop_out(h0, 0.2, True) # [16, 16]

            h1 = self._conv_cond_concat(h0, yb)

            h1 = self._conv2d(h1, 64, k_h=3, k_w=3, d_h=1, d_w=1, kernel_initializer=he_init, name="conv2d_10")
            h1 = self.leakyReLu(h1)
            h1 = self._conv_cond_concat(h1, yb)

            h1 = self._conv2d(h1, 64, k_h=3, k_w=3, d_h=2, d_w=2, kernel_initializer=he_init, name="conv2d_11")
            h1 = self.leakyReLu(h1)
            h1 = self._drop_out(h1, 0.2, True) # [8, 8]

            h2 = self._conv_cond_concat(h1, yb)
            h2 = self._conv2d(h2, 128, k_h=3, k_w=3, d_h=1, d_w=1, kernel_initializer=he_init, name="conv2d_20")
            h2 = self.leakyReLu(h2)
            h2 = self._conv_cond_concat(h2, yb)

            # h2 = self._conv_cond_concat(h2, yb)
            h2 = self._conv2d(h2, 128, k_h=3, k_w=3, d_h=1, d_w=1, kernel_initializer=he_init, name="conv2d_21")
            h2 = self.leakyReLu(h2)

            h3 = tf.layers.average_pooling2d(h2, pool_size = 8, strides=1,
                                        name='avg_pool_0')
            h3 = tf.squeeze(h3, [1, 2])
            h3 = tf.concat([h3, y], 1)
            h3 = self._linear_fc(h3, 1, 'lin', kernel_initializer = he_init)
        return tf.nn.sigmoid(h3), h3

    def classifier(self, inp, is_training, init=False, reuse=False, getter=None):
        with tf.variable_scope('classifier', reuse=reuse, custom_getter=getter):
            x = tf.reshape(inp, [-1, 32, 32, 3])
            x = self._add_noise(x, stddev=0.15)

            x = nn.conv2d_WN(x, num_filters=128, init = init, use_weight_normalization = True,
                            use_batch_normalization = False,
                            use_mean_only_batch_normalization = True,
                            deterministic = tf.math.logical_not(is_training),
                            name = 'conv1_1', nonlinearity = self.leakyReLu)
            x = nn.conv2d_WN(x, num_filters=128, init=init, use_weight_normalization=True,
                             use_batch_normalization=False,
                             use_mean_only_batch_normalization=True,
                             deterministic=tf.math.logical_not(is_training),
                             name='conv1_2', nonlinearity=self.leakyReLu)
            x = nn.conv2d_WN(x, num_filters=128, init=init, use_weight_normalization=True,
                             use_batch_normalization=False,
                             use_mean_only_batch_normalization=True,
                             deterministic=tf.math.logical_not(is_training),
                             name='conv1_3', nonlinearity=self.leakyReLu)


            x = tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'max_pool_1')
            x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_1')

            x = nn.conv2d_WN(x, num_filters=256, init=init, use_weight_normalization=True,
                             use_batch_normalization=False,
                             use_mean_only_batch_normalization=True,
                             deterministic=tf.math.logical_not(is_training),
                             name='conv2_1', nonlinearity=self.leakyReLu)
            x = nn.conv2d_WN(x, num_filters=256, init=init, use_weight_normalization=True,
                             use_batch_normalization=False,
                             use_mean_only_batch_normalization=True,
                             deterministic=tf.math.logical_not(is_training),
                             name='conv2_2', nonlinearity=self.leakyReLu)
            x = nn.conv2d_WN(x, num_filters=256, init=init, use_weight_normalization=True,
                             use_batch_normalization=False,
                             use_mean_only_batch_normalization=True,
                             deterministic=tf.math.logical_not(is_training),
                             name='conv2_3', nonlinearity=self.leakyReLu)

            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name ='max_pool_2')
            x = tf.layers.dropout(x, rate=0.5, training=is_training, name='dropout_2')

            x = nn.conv2d_WN(x, num_filters=512, init=init, use_weight_normalization=True,
                             use_batch_normalization=False,
                             use_mean_only_batch_normalization=True,
                             deterministic=tf.math.logical_not(is_training),
                             name='conv3', nonlinearity = self.leakyReLu, pad='VALID')

            x = nn.NiN_WN(x, num_units = 256, nonlinearity = self.leakyReLu, init=init,
                       use_weight_normalization = True,
                       use_batch_normalization = False,
                       use_mean_only_batch_normalization = True,
                       deterministic = tf.math.logical_not(is_training), name='NiN1')

            x = nn.NiN_WN(x, num_units = 128, nonlinearity=self.leakyReLu, init=init,
                       use_weight_normalization=True,
                       use_batch_normalization=False,
                       use_mean_only_batch_normalization=True,
                       deterministic=tf.math.logical_not(is_training), name='NiN2')

            x = tf.layers.max_pooling2d(x, pool_size=6, strides=1,
                                        name='avg_pool_0')
            x = tf.squeeze(x, [1, 2])

            intermediate_layer = x

            logits = nn.dense_WN(x, num_units = 10, nonlinearity = None, init=init, use_weight_normalization = True,
                                 use_batch_normalization = False,
                                 use_mean_only_batch_normalization = True,
                                 deterministic = tf.math.logical_not(is_training), name='output_dense')

        return logits, intermediate_layer

    def good_sampler(self, z, y):
        with tf.variable_scope('good_generator', reuse=True):
            counter = {}
            # project 'z' and reshape
            yb = tf.reshape(y, [tf.shape(y)[0], 1, 1, self.config.NUM_CLASSES])
            z = tf.concat([z, y], 1)

            h0 = self._linear_fc(z, 4 * 4 * 512, 'gg_h0_lin', kernel_initializer=he_init)
            h0 = tf.nn.relu(h0, 'gg_rl0')  # [4,4]
            h0 = self._batch_norm_contrib(h0, 'gg_bn0', train=True)
            h0 = tf.reshape(h0, [-1, 4, 4, 512])
            h0 = self._conv_cond_concat(h0, yb)

            h0 = self._deconv2d(h0, 256, k_w=5, k_h=5, d_w=2, d_h=2, name='gg_dconv0',
                                kernel_initializer=he_init)  # [8, 8]
            h0 = tf.nn.relu(h0, 'gg_rl1')
            h0 = self._batch_norm_contrib(h0, 'gg_bn1', train=True)
            h0 = self._conv_cond_concat(h0, yb)

            h1 = self._deconv2d(h0, 128, k_w=5, k_h=5, d_w=2, d_h=2, name='gg_dconv1', kernel_initializer=he_init)
            h1 = tf.nn.relu(h1, 'gg_rl2')  # [16,16]
            h1 = self._batch_norm_contrib(h1, 'gg_bn2', train=True)
            h1 = self._conv_cond_concat(h1, yb)

            h2 = self._deconv2d(h1, 3, k_w=5, k_h=5, d_w=2, d_h=2, name='gg_dconv2', kernel_initializer=he_init)
            h2 = tf.nn.tanh(h2)
        return h2

    def forward_pass(self, z_g, y_g, x_l_c, y_l_c, x_l_d, y_l_d, x_u_d, x_u_c, train):
        """
        :param z: latent variable [200, 100]
        :param x_l_c: [200, 32, 32, 3]
        :param y_l_c: [200, 10]
        :param x_l_d: [20, 32, 32, 3]
        :param y_l_d: [20, 10]
        :param x_u_d: [180, 32, 32, 3]
        :param x_u_c: [200, 32, 32, 3]
        :return:
        """
        # output of G
        self.good_generator(z_g, y_g, init = True, reuse = False)  # init of weightnorm weights cf Salimans et al.
        G = self.good_generator(z_g, y_g, init = False, reuse = True)

        if self.config.DATA_NAME == "cifar10":
            ## apply ZCA:
            whitenner = cifar10_ZCA(self.config)
            x_u_c_zca = whitenner.apply(x_u_c)
            x_l_c_zca = whitenner.apply(x_l_c)
            x_u_d_zca = whitenner.apply(x_u_d)
            G_zca = whitenner.apply(G)
            # output of C for real images
            self.classifier(x_u_c_zca, train, init=True, reuse=False)  # init of weightnorm weights cf Salimans et al.
            C_real_logits, _ = self.classifier(x_l_c_zca, train, init=False, reuse=True)

            # output of C for unlabel images (as false examples to D)
            C_unl_logits, _ = self.classifier(x_u_c_zca, train, init=False, reuse=True)
            C_unl_hard = tf.argmax(C_unl_logits, axis=1)
            C_unl_logits_rep, _ = self.classifier(x_u_c_zca, train, init=False, reuse=True)

            # output of C for unlabel images (as positive examples to C)
            C_unl_d_logits, _ = self.classifier(x_u_d_zca, train, init=False, reuse=True)
            C_unl_d_hard = tf.argmax(C_unl_d_logits, axis=1)

            # output of C for generated images
            C_fake_logits, _ = self.classifier(G_zca, train, init=False, reuse=True)
        else:
            # output of C for real images
            self.classifier(x_u_c, train, init = True, reuse = False) # init of weightnorm weights cf Salimans et al.
            C_real_logits, _ = self.classifier(x_l_c, train, init = False, reuse = True)

            # output of C for unlabel images (as false examples to D)
            C_unl_logits, _ = self.classifier(x_u_c, train, init = False, reuse = True)
            C_unl_hard = tf.argmax(C_unl_logits, axis = 1)

            # output of C for unlabel images (as positive examples to C)
            C_unl_d_logits, _ = self.classifier(x_u_d, train, init = False, reuse = True)
            C_unl_d_hard = tf.argmax(C_unl_d_logits, axis = 1)

            # output of C for generated images
            C_fake_logits, _ = self.classifier(G, train, init = False, reuse = True)

        # output of D for positive examples
        X_P = tf.concat([x_l_d, x_u_d], axis = 0)
        Y_P = tf.concat([y_l_d, tf.one_hot(C_unl_d_hard, depth = self.config.NUM_CLASSES)], axis = 0)
        X_P.set_shape([self.config.BATCH_SIZE_L_D + self.config.BATCH_SIZE_U_D] + self.config.IMAGE_DIM)
        Y_P.set_shape([self.config.BATCH_SIZE_L_D + self.config.BATCH_SIZE_U_D, self.config.NUM_CLASSES])

        self.discriminator(X_P, Y_P, init = True, reuse = False) # init of weightnorm weights cf Salimans et al.
        D_real, D_real_logits = self.discriminator(X_P, Y_P, init = False, reuse = True)

        # output of D for generated examples
        D_fake, D_fake_logits  = self.discriminator(G, y_g, init = False, reuse = True)

        # output of D for unlabeled examples (negative example)
        D_unl, D_unl_logits  = self.discriminator(x_u_c, tf.one_hot(C_unl_hard, depth = self.config.NUM_CLASSES),\
                                                         init = False, reuse = True)

        if self.config.DATA_NAME == "cifar10":
            return [G, [D_real, D_real_logits, D_fake, D_fake_logits, D_unl, D_unl_logits],
                    [C_real_logits, C_unl_logits, C_unl_d_logits, C_fake_logits, C_unl_logits_rep]]
        else:
            return [G, [D_real, D_real_logits, D_fake, D_fake_logits, D_unl, D_unl_logits],
                    [C_real_logits, C_unl_logits, C_unl_d_logits, C_fake_logits]]

    def forward_pass_CGAN(self, z, image, y):
        G = self.good_generator(z, y, reuse = False)
        D_real, D_real_logits, fm_real = self.discriminator(image, y, reuse = False)
        D_fake, D_fake_logits, fm_fake = self.discriminator(G, y, reuse = True)
        D = [D_real, D_real_logits, fm_real, D_fake, D_fake_logits, fm_fake]
        return G, D

class cifar10_ZCA():
    def __init__(self, config):
        m = np.load(os.path.join(config.DATA_DIR, "cifar10_zca_mean.npy"))
        mat = np.load(os.path.join(config.DATA_DIR, "cifar10_zca_mat.npy"))
        self.cifar_zca_mean = tf.constant(m)
        self.cifar_zca_mat = tf.constant(mat)

    def apply(self, image):
        s = tf.shape(image)
        image = tf.reshape(tf.matmul((tf.reshape(image, [s[0], s[1] * s[2] * s[3]]) - self.cifar_zca_mean), self.cifar_zca_mat),
                           (-1, s[1], s[2], s[3]))

        return image

if __name__ == "__main__":
    from config import Config

    class TempConfig(Config):
        DATA_NAME = "svhn"
        NUM_CLASSES = 10
        MINIBATCH_DIS = False
        BATCH_SIZE = 100
        BATCH_SIZE_L_D = 20
        BATCH_SIZE_U_D = 200


    tmpconfig = TempConfig()

    tf.reset_default_graph()

    image = tf.ones((64, 32, 32, 3))
    y = tf.ones((64, 10))
    z = tf.ones((64, 100))
    model = Good_GAN_simple(tmpconfig)

    G, C = model.forward_pass(z, y, image, y, image, y, image, image, True)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        h_o = sess.run(G + C)

    print(len(h_0))
