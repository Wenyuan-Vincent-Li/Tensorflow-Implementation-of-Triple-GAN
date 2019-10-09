'''
This is a python file that used for training GAN.
TODO: provide a parser access from terminal.
'''
## Import module
import sys
import os
if os.getcwd().endswith("Path_Semi_GAN"):
    root_dir = os.getcwd()
else:
    root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import numpy as np
import tensorflow as tf
from time import strftime
import math
from datetime import datetime
from pytz import timezone

from Training.train_base import Train_base
from Training.Saver import Saver
from Training.Summary import Summary
from utils import *
from tensorflow.python import debug as tf_debug

class Train(Train_base):
    def __init__(self, config, log_dir, save_dir, **kwargs):
        # Reset tf graph.
        tf.reset_default_graph()
        self.lr_ph = tf.placeholder(tf.float32, name='learning_rate')
        self.cla_lr_ph = tf.placeholder(tf.float32, name='cla_learning_rate')
        # self.cla_beta1 = tf.placeholder(tf.float32, shape = [], name='cla_beta1')
        super(Train, self).__init__()
        self.config = config
        self.save_dir = save_dir
        self.comments = kwargs.get('comments', '')
        if self.config.SUMMARY:
            self.summary_train = Summary(log_dir, config, log_type='train', \
                                         log_comments=kwargs.get('comments', ''))
            self.summary_val = Summary(log_dir, config, log_type='val', \
                                       log_comments=kwargs.get('comments', ''))

    def train(self, Dataset, Model, sample_y):
        # Create input node
        init_op_train, init_op_val, NNIO = self._input_fn_train_val(Dataset)
        x_l_c, y_l_c, x_l_d, y_l_d, x_u = NNIO

        with tf.device('/gpu:0'):
            # Build up the graph
            PH, G, D, C, model = self._build_train_graph(Model)

            z_g_ph, y_g_ph, x_l_c_ph, y_l_c_ph, x_l_d_ph, y_l_d_ph, x_u_d_ph, x_u_c_ph, \
            train_ph, Lambda = PH
            if self.config.DATA_NAME == "cifar10":
                C_real_logits, C_unl_logits, C_unl_d_logits, C_fake_logits, C_unl_logits_rep = C
            else:
                C_real_logits, C_unl_logits, C_unl_d_logits, C_fake_logits = C

            lambda_1_ph = Lambda[0]
            if self.config.DATA_NAME == "cifar10":
                lambda_2_ph = Lambda[1]

            ## Sample the images
            sample_z_ph = tf.placeholder(tf.float32, [self.config.SAMPLE_SIZE, self.config.Z_DIM],
                           name='sample_latent_variable')  # latent variable
            sample_y_ph = tf.placeholder(tf.float32, [self.config.SAMPLE_SIZE, self.config.NUM_CLASSES],
                                    name='condition_label_for_sampler')
            samples = model.good_sampler(sample_z_ph, sample_y_ph)

            # Create Loss function
            d_loss, g_loss, c_loss = self._goodGAN_loss(G, D, C, None, [y_g_ph, y_l_c_ph], Lambda, model.discriminator)


            # Create the metric
            accuracy, update_op, reset_op, preds, probs = self._metric(C_real_logits, y_l_c_ph)

        # Create optimizer
        with tf.name_scope('Train'):
            t_vars = tf.trainable_variables()

            g_vars = [var for var in t_vars if 'good_generator' in var.name]
            d_vars = [var for var in t_vars if 'discriminator' in var.name]
            c_vars = [var for var in t_vars if 'classifier' in var.name]

            d_optimizer = self._Adam_optimizer(lr = self.lr_ph, beta1 = self.config.BETA1)
            g_optimizer = self._Adam_optimizer(lr = self.lr_ph, beta1 = self.config.BETA1)
            c_optimizer = self._Adam_optimizer(lr = self.cla_lr_ph, beta1 = 0.5)
            # optimizer = self._Adam_optimizer(self.lr_ph, self.config.BETA1)
            d_solver = self._train_op(d_optimizer, d_loss, \
                                      var_list = d_vars)
            g_solver = self._train_op(g_optimizer, g_loss, \
                                       var_list = g_vars)
            c_solver_ = self._train_op(c_optimizer, c_loss,
                                      var_list = c_vars)

            ## add weight normalization for classifier
            # c_weight_loss = self._loss_weight_l2([var for var in t_vars if 'c_h2_lin' in var.name and 'b' not in var.name], eta = 1e-4)
            # c_loss += c_weight_loss

            # add moving average
            ema = tf.train.ExponentialMovingAverage(decay = 0.9999)
            with tf.control_dependencies([c_solver_]):
                c_solver = ema.apply(c_vars)

        # Add summary
        if self.config.SUMMARY:
            summary_dict_train = {}
            summary_dict_val = {}
            if self.config.SUMMARY_SCALAR:
                scalar_train = {'g_loss': g_loss, 'd_loss': d_loss, 'c_loss': c_loss,
                                'train_accuracy': accuracy}
                scalar_val = {'val_accuracy': accuracy}
                summary_dict_train['scalar'] = scalar_train
                summary_dict_val['scalar'] = scalar_val

            # Merge summary
            merged_summary_train = \
                self.summary_train.add_summary(summary_dict_train)
            merged_summary_val = \
                self.summary_val.add_summary(summary_dict_val)

        # Add saver
        saver = Saver(self.save_dir)

        # Create a session for training
        sess_config = tf.ConfigProto(allow_soft_placement = True, gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8))
        # Use soft_placement to place those variables, which can be placed, on GPU

        # Build up latent variable for samples
        sample_z = np.random.uniform(low = -1.0, high = 1.0, size = (self.config.SAMPLE_SIZE, self.config.Z_DIM)).astype(np.float32)

        with tf.Session(config = sess_config) as sess:
            if self.config.DEBUG:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # Add graph to tensorboard
            if self.config.SUMMARY and self.config.SUMMARY_GRAPH:
                self.summary_train._graph_summary(sess.graph)

            lr = self.config.LEARNING_RATE
            cla_lr = self.config.CLA_LEARNINIG_RATE
            # Restore teh weights from the previous training
            if self.config.RESTORE:
                start_epoch = saver.restore(sess, dir_names=self.config.RUN, epoch=self.config.RESTORE_EPOCH)
                if start_epoch >= 300:
                    lr = lr * 0.995 ** (start_epoch - 300)
                    cla_lr = cla_lr * 0.99 ** (start_epoch - 300)
                initialize_uninitialized_vars(sess)
            else:
                # Create a new folder for saving model
                saver.set_save_path(comments=self.comments)
                start_epoch = 0

                # initialize the variables
                init_var = tf.group(tf.global_variables_initializer(), \
                                    tf.local_variables_initializer())
                sess.run(init_var)

            # Start Training
            tf.logging.info("Start training!")
            for epoch in range(1, self.config.EPOCHS + 1):
                tf.logging.info("Training for epoch {}.".format(epoch + start_epoch))
                train_pr_bar = tf.contrib.keras.utils.Progbar(target= \
                                                                  int(self.config.TRAIN_SIZE / self.config.BATCH_SIZE))

                lambda_1 = self.config.FAKE_G_LAMBDA if (start_epoch + epoch) > 200 else 0.

                if self.config.DATA_NAME == "cifar10":
                    # rampup_value = rampup(start_epoch + epoch)
                    # rampdown_value = rampdown(start_epoch + epoch)
                    # lambda_2 = rampup_value * 20 if epoch > 1 else 0
                    lambda_2 = 0.5 if epoch > 67 else 0
                    # b1_c = rampdown_value * 0.9 + (1.0 - rampdown_value) * 0.5
                    # lambda_2 = 0

                if (start_epoch + epoch >= 300):
                    lr = lr * 0.995
                    cla_lr = cla_lr * 0.99

                tf.logging.info("Lambda_1 {}.".format(lambda_1))
                sess.run(init_op_train)

                if self.config.PRE_TRAIN and (start_epoch + epoch <= 30):
                    tf.logging.info("Pre Training!")
                    for i in range(int(self.config.TRAIN_SIZE / self.config.BATCH_SIZE)):
                        # Feature labeled and unlabeled data
                        x_l_c_o, y_l_c_o, x_l_d_o, y_l_d_o, x_u_o = sess.run([x_l_c, y_l_c, x_l_d, y_l_d, x_u])
                        # Define latent vector
                        z = np.random.uniform(low = -1.0, high = 1.0, size=(self.config.BATCH_SIZE_G, self.config.Z_DIM)).astype(np.float32)
                        # y = y_l_c_o  ## Todo: think about if this is the best way
                        y_temp = np.random.randint(low = 0, high = self.config.NUM_CLASSES, size = (self.config.BATCH_SIZE_G))
                        y = np.zeros((self.config.BATCH_SIZE_G, self.config.NUM_CLASSES))
                        y[np.arange(self.config.BATCH_SIZE_G), y_temp] = 1

                        if self.config.DATA_NAME == "cifar10":
                            z_g_ph, y_g_ph, x_l_c_ph, y_l_c_ph, x_l_d_ph, y_l_d_ph, x_u_d_ph, x_u_c_ph, \
                            train_ph, Lambda_ph = PH
                            lambda_1_ph, lambda_2_ph = Lambda_ph
                        else:
                            z_g_ph, y_g_ph, x_l_c_ph, y_l_c_ph, x_l_d_ph, y_l_d_ph, x_u_d_ph, x_u_c_ph, \
                            train_ph, lambda_1_ph = PH
                            lambda_1_ph = lambda_1_ph[0]

                        feed_dict = {z_g_ph: z,
                                     y_g_ph: y,
                                     x_l_c_ph: x_l_c_o,
                                     y_l_c_ph: y_l_c_o,
                                     x_l_d_ph: x_l_d_o,
                                     y_l_d_ph: y_l_d_o,
                                     x_u_d_ph: x_u_o[:self.config.BATCH_SIZE_U_D, ...],
                                     x_u_c_ph: x_u_o[self.config.BATCH_SIZE_U_D : self.config.BATCH_SIZE_U_D + self.config.BATCH_SIZE_U_C, ...],
                                     train_ph: True,
                                     lambda_1_ph: lambda_1,
                                     self.lr_ph: lr,
                                     self.cla_lr_ph: cla_lr}

                        if self.config.DATA_NAME == "cifar10":
                            feed_dict[lambda_2_ph] = lambda_2
                            # feed_dict[self.cla_beta1] = b1_c


                        # Update classifier
                        _, c_loss_o = sess.run([c_solver, c_loss], \
                                               feed_dict=feed_dict)
                        # Update progress bar
                        train_pr_bar.update(i)


                else:
                    tf.logging.info("Good-GAN Training!")
                    for i in range(int(self.config.TRAIN_SIZE / self.config.BATCH_SIZE)):
                        # Feature labeled and unlabeled data
                        x_l_c_o, y_l_c_o, x_l_d_o, y_l_d_o, x_u_o = sess.run([x_l_c, y_l_c, x_l_d, y_l_d, x_u])
                        # Define latent vector
                        z = np.random.uniform(low = -1.0, high = 1.0, size = (self.config.BATCH_SIZE_G, self.config.Z_DIM)).astype(np.float32)

                        y_temp = np.random.randint(low=0, high=self.config.NUM_CLASSES,
                                                   size=(self.config.BATCH_SIZE_G))
                        y = np.zeros((self.config.BATCH_SIZE_G, self.config.NUM_CLASSES))
                        y[np.arange(self.config.BATCH_SIZE_G), y_temp] = 1

                        if self.config.DATA_NAME == "cifar10":
                            z_g_ph, y_g_ph, x_l_c_ph, y_l_c_ph, x_l_d_ph, y_l_d_ph, x_u_d_ph, x_u_c_ph, \
                            train_ph, Lambda_ph = PH
                            lambda_1_ph, lambda_2_ph = Lambda_ph
                        else:
                            z_g_ph, y_g_ph, x_l_c_ph, y_l_c_ph, x_l_d_ph, y_l_d_ph, x_u_d_ph, x_u_c_ph, \
                            train_ph, lambda_1_ph = PH
                            lambda_1_ph = lambda_1_ph[0]
                        feed_dict = {z_g_ph: z,
                                     y_g_ph: y,
                                     x_l_c_ph: x_l_c_o,
                                     y_l_c_ph: y_l_c_o,
                                     x_l_d_ph: x_l_d_o,
                                     y_l_d_ph: y_l_d_o,
                                     x_u_d_ph: x_u_o[:self.config.BATCH_SIZE_U_D,...],
                                     x_u_c_ph: x_u_o[self.config.BATCH_SIZE_U_D : self.config.BATCH_SIZE_U_D + self.config.BATCH_SIZE_U_C,...],
                                     train_ph: True,
                                     lambda_1_ph: lambda_1,
                                     self.lr_ph: lr,
                                     self.cla_lr_ph: cla_lr}

                        if self.config.DATA_NAME == "cifar10":
                            feed_dict[lambda_2_ph] = lambda_2
                            # feed_dict[self.cla_beta1] = b1_c

                        # Update discriminator
                        _, d_loss_o = sess.run([d_solver, d_loss], \
                                   feed_dict = feed_dict)
                        # Update generator
                        _, g_loss_o = sess.run([g_solver, g_loss], \
                                    feed_dict = feed_dict)
                        # _, g_loss_o = sess.run([g_solver, g_loss], \
                        #                        feed_dict=feed_dict)
                        # Update classifier
                        _, c_loss_o = sess.run([c_solver, c_loss], \
                                    feed_dict = feed_dict)
                        # Update progress bar
                        train_pr_bar.update(i)

                # Get the training statistics
                summary_train_o, d_loss_o, g_loss_o, c_loss_o = sess.run([merged_summary_train,
                                                                                       d_loss,
                                                                                       g_loss,
                                                                                       c_loss],
                                                                                     feed_dict = feed_dict)

                tf.logging.info("The training stats for epoch {}: g_loss: {:.2f}, d_loss {:.2f}, c_loss: {:.2f}."\
                                .format(epoch + start_epoch, g_loss_o, d_loss_o, c_loss_o))


                if self.config.SUMMARY:
                    # Add summary
                    self.summary_train.summary_writer.add_summary(summary_train_o, epoch + start_epoch)

                # Perform validation
                tf.logging.info("\nValidate for epoch {}.".format(epoch + start_epoch))
                sess.run(init_op_val + [reset_op])
                if not self.config.VAL_STEP: # perform full validation
                    while True:
                        try:
                            x_l_c_o, y_l_c_o = sess.run([x_l_c, y_l_c])
                            feed_dict = {z_g_ph: z,
                                         y_g_ph: y,
                                         x_l_c_ph: x_l_c_o,
                                         y_l_c_ph: y_l_c_o,
                                         x_l_d_ph: x_l_d_o,
                                         y_l_d_ph: y_l_d_o,
                                         x_u_d_ph: x_u_o[:self.config.BATCH_SIZE_U_D,...],
                                         x_u_c_ph: x_u_o[self.config.BATCH_SIZE_U_D : self.config.BATCH_SIZE_U_D + self.config.BATCH_SIZE_U_C,...],
                                         train_ph: False,
                                         lambda_1_ph: lambda_1,
                                         self.lr_ph: lr,
                                         self.cla_lr_ph: cla_lr}

                            if self.config.DATA_NAME == "cifar10":
                                feed_dict[lambda_2_ph] = lambda_2
                                # feed_dict[self.cla_beta1] = b1_c

                            accuracy_o, summary_val_o, _ = sess.run([accuracy, merged_summary_val] + update_op, \
                                                                    feed_dict = feed_dict)
                        except (tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError, ValueError):
                            break
                else:
                    for i in range(self.config.VAL_STEP):
                        x_l_c_o, y_l_c_o = sess.run([x_l_c, y_l_c])
                        feed_dict = {z_g_ph: z,
                                     y_g_ph: y,
                                     x_l_c_ph: x_l_c_o,
                                     y_l_c_ph: y_l_c_o,
                                     x_l_d_ph: x_l_d_o,
                                     y_l_d_ph: y_l_d_o,
                                     x_u_d_ph: x_u_o[:self.config.BATCH_SIZE_U_D, ...],
                                     x_u_c_ph: x_u_o[self.config.BATCH_SIZE_U_D : self.config.BATCH_SIZE_U_D + self.config.BATCH_SIZE_U_C, ...],
                                     train_ph: False,
                                     lambda_1_ph: lambda_1,
                                     self.lr_ph: lr,
                                     self.cla_lr_ph: cla_lr}

                        if self.config.DATA_NAME == "cifar10":
                            feed_dict[lambda_2_ph] = lambda_2
                            # feed_dict[self.cla_beta1] = b1_c

                        accuracy_o, summary_val_o, _ = sess.run([accuracy, merged_summary_val] + update_op, \
                                                                feed_dict=feed_dict)

                tf.logging.info(
                    "\nThe current validation accuracy for epoch {} is {:.2f}.\n" \
                    .format(epoch + start_epoch, accuracy_o))
                # Add summary to tensorboard
                if self.config.SUMMARY:
                    self.summary_val.summary_writer.add_summary(summary_val_o, epoch + start_epoch)

                # Sample generated images
                samples_o = sess.run(samples,
                                     feed_dict = {
                                         sample_z_ph: sample_z,
                                         sample_y_ph: sample_y
                                     })

                if self.config.DATA_NAME == "mnist":
                    samples_o = np.reshape(samples_o, [-1, 28, 28, 1])

                save_images(samples_o[:64], image_manifold_size(64), \
                            os.path.join(self.config.SAMPLE_DIR, 'train_{:02d}.png'.format(epoch + start_epoch)))

                # Save the model per SAVE_PER_EPOCH
                if epoch % self.config.SAVE_PER_EPOCH == 0:
                    save_name = str(epoch + start_epoch)
                    saver.save(sess, 'model_' + save_name.zfill(4) \
                               + '.ckpt')

            if self.config.SUMMARY:
                self.summary_train.summary_writer.flush()
                self.summary_train.summary_writer.close()
                self.summary_val.summary_writer.flush()
                self.summary_val.summary_writer.close()

                # Save the model after all epochs
            save_name = str(epoch + start_epoch)
            saver.save(sess, 'model_' + save_name.zfill(4) + '.ckpt')
        return

    def _input_fn_train_val(self, Dataset):
        '''
        Create the input node.
        '''
        with tf.device('/cpu:0'):
            with tf.name_scope('Input_Data'):
                # Training dataset
                dataset_train = Dataset(self.config.DATA_DIR, self.config, \
                                        self.config.NUM_LABEL, 'train', True)
                # Validation dataset
                dataset_val = Dataset(self.config.DATA_DIR, self.config, \
                                      self.config.NUM_LABEL, 'test', False)
                # Inputpipeline
                init_op_train, init_op_val, NNIO = dataset_train.inputpipline_train_val(
                    dataset_val)
        return init_op_train, init_op_val, NNIO

    def _build_train_graph(self, Model):
        '''
            Build up the training graph.
        '''
        # Create the input placeholder
        z_g_ph = tf.placeholder(tf.float32, [self.config.BATCH_SIZE_G, self.config.Z_DIM], name = 'latent_variable')  # latent variable
        y_g_ph = tf.placeholder(tf.float32, [self.config.BATCH_SIZE_G, self.config.NUM_CLASSES], name='condition_label_for_g')
        x_l_c_ph = tf.placeholder(tf.float32, [self.config.BATCH_SIZE_L_C] + self.config.IMAGE_DIM, name='labeled_images_for_c')
        y_l_c_ph = tf.placeholder(tf.float32, [self.config.BATCH_SIZE_L_C, self.config.NUM_CLASSES], name='real_label_for_c')
        x_l_d_ph = tf.placeholder(tf.float32, [self.config.BATCH_SIZE_L_D] + self.config.IMAGE_DIM, name='labeled_images_for_d')
        y_l_d_ph = tf.placeholder(tf.float32, [self.config.BATCH_SIZE_L_D, self.config.NUM_CLASSES],
                                  name='real_label_for_d')
        x_u_d_ph = tf.placeholder(tf.float32, [self.config.BATCH_SIZE_U_D] + self.config.IMAGE_DIM, name='unlabeled_images_for_d')
        x_u_c_ph = tf.placeholder(tf.float32, [self.config.BATCH_SIZE_U_C] + self.config.IMAGE_DIM, name='unlabeled_images_for_c')

        train_ph = tf.placeholder(tf.bool, name = "is_training") # boolean variable: true for training and false for inference
        lambda_1_ph = tf.placeholder(tf.float32, name = 'lambda_fake')  # lamda_1 for loss function
        Lambda = [lambda_1_ph]
        if self.config.DATA_NAME == "cifar10":
            lambda_2_ph = tf.placeholder(tf.float32, name = 'lambda_unsup')  # lamda_1 for loss function
            Lambda += [lambda_2_ph]
        # Create the model
        main_graph = Model(self.config)
        G, D, C = main_graph.forward_pass(z_g_ph, y_g_ph, x_l_c_ph, y_l_c_ph, x_l_d_ph, y_l_d_ph, x_u_d_ph, x_u_c_ph, train_ph)

        return [z_g_ph, y_g_ph, x_l_c_ph, y_l_c_ph, x_l_d_ph, y_l_d_ph, x_u_d_ph, x_u_c_ph, \
                train_ph, Lambda], G, D, C, main_graph

    def _metric(self, real_lab_logits, real_lab):
        '''
        Create evaluation metric.
        '''
        with tf.name_scope('Metric') as scope:
            real_lab = tf.argmax(real_lab, axis=-1)
            prediction = tf.argmax(real_lab_logits, axis=-1)
            probs = tf.nn.softmax(real_lab_logits)[:, -1]

            # Prediction accuracy
            accuracy, update_op_a = self._accuracy_metric(real_lab, prediction)

            # Update op inside each validation run
            update_op = [update_op_a]

            # Reset op for each validation run
            variables = tf.contrib.framework.get_variables(
                scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            reset_op = tf.variables_initializer(variables)
        return accuracy, update_op, reset_op, prediction, probs

    def _goodGAN_loss(self, G, D, C, X, Y, Lambda, discriminator = None):
        '''
            Create loss function.
        '''
        d_loss, g_loss, c_loss = self._loss_GAN(D, C, Y, Lambda)
        return d_loss, g_loss, c_loss

def rampup(epoch):
    if epoch < 300:
        p = max(0.0, float(epoch)) / float(300)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def rampdown(epoch):
    if epoch >= (300 - 50):
        ep = (epoch - (300 - 50)) * 0.5
        return math.exp(-(ep * ep) / 50)
    else:
        return 1.0


def _main_training_svhn(FLAGS = None):
    from config import Config
    from Input_Pipeline.svhnDataset import svhnDataset as Dataset

    if FLAGS is not None and FLAGS.simple:
        from Model.Good_GAN import Good_GAN as Model
    else:
        from Model.Good_GAN import Good_GAN as Model

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class TempConfig(Config):
        NAME = "Good_GAN"
        ## Input pipeline
        DATA_NAME = "svhn"
        DATA_DIR = os.path.join(root_dir, "DataSet/svhn")
        NUM_LABEL = 500
        BATCH_SIZE = 100
        BATCH_SIZE_G = BATCH_SIZE
        BATCH_SIZE_bG = 20
        BATCH_SIZE_L_C = 50
        BATCH_SIZE_U_C = 50
        BATCH_SIZE_L_D = 20
        BATCH_SIZE_U_D = 80


        IMAGE_HEIGHT = 32
        IMAGE_WIDTH = 32
        CHANNEL = 3
        REPEAT = -1

        FAKE_G_LAMBDA = 0.03
        CLA_LEARNINIG_RATE = 3e-4

        ## Model architecture
        # Number of classification classes
        Z_DIM = 100
        NUM_CLASSES = 10
        MINIBATCH_DIS = False

        ## Traning settings
        # Restore
        RESTORE = False  # Whether to use the previous trained weights
        # RUN = "Run_2019-04-24_22_29_06" ## run for svhn_200
        # RUN = "Run_2019-04-25_15_04_01" ## run for svhn 250
        # RESTORE_EPOCH = 200
        LEARNING_RATE = 3e-4

        # Training schedule
        EPOCHS = 1000
        TRAIN_SIZE = 73257 - NUM_LABEL  # Num of unlabeled data
        SAVE_PER_EPOCH = 1 # epochs to save trained_weights
        VAL_STEP = None

        # Results
        SAMPLE_DIR = "good_GAN_svhn_500"
        WEIGHT_DIR = os.path.join(root_dir, "Training/Weight_svhn")
        LOG_DIR = os.path.join(root_dir, "Training/Log_svhn")

    # Create the global configuration
    tmp_config = TempConfig()
    if FLAGS:
        _customize_config(tmp_config, FLAGS)
    tmp_config.SAMPLE_DIR = os.path.join(root_dir, "Training", tmp_config.SAMPLE_DIR)
    if tmp_config.NUM_LABEL < 1000:
        tmp_config.PRE_TRAIN = True
    tmp_config.display()
    check_folder_exists(tmp_config.SAMPLE_DIR)

    # Comments log on the current run
    comments = "This training is for svhn dataset."
    comments += tmp_config.config_str() + datetime.now(timezone('America/Los_Angeles')).strftime("%Y-%m-%d_%H_%M_%S")
    # Create a training object
    training = Train(tmp_config, tmp_config.LOG_DIR, tmp_config.WEIGHT_DIR, comments=comments)
    # Load in sample_y
    sample_y = np.load(os.path.join(root_dir, "DataSet/svhn/svhn_sample_y.npy"))
    training.train(Dataset, Model, sample_y)

def _main_training_cifar10(FLAGS = None):
    from config import Config
    from Input_Pipeline.cifar10Dataset import cifar10Dataset as Dataset
    # from Model.Good_GAN_simple import Good_GAN_simple as Model
    from Model.Good_GAN_cifar10 import Good_GAN_cifar10 as Model

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class TempConfig(Config):
        NAME = "Good_GAN"
        ## Input pipeline
        DATA_NAME = "cifar10"
        DATA_DIR = os.path.join(root_dir, "DataSet/cifar_10")
        NUM_LABEL = 4000
        BATCH_SIZE_G = 100
        BATCH_SIZE_bG = 10
        BATCH_SIZE_L_C = 50
        BATCH_SIZE_U_C = 50
        BATCH_SIZE_L_D = 20
        BATCH_SIZE_U_D = 80
        BATCH_SIZE = BATCH_SIZE_G

        IMAGE_HEIGHT = 32
        IMAGE_WIDTH = 32
        CHANNEL = 3
        REPEAT = -1

        ### TODO: check this number
        FAKE_G_LAMBDA = 0.3

        ## Model architecture
        # Number of classification classes
        Z_DIM = 100
        NUM_CLASSES = 10
        MINIBATCH_DIS = False

        ## Traning settings
        # Restore
        RESTORE = True  # Whether to use the previous trained weights
        # RUN = "Run_2019-04-11_01_46_38" # deepmed unsup
        RUN = "Run_2019-05-08_21_28_34" ## Local triple_tf
        # RESTORE_EPOCH = 200
        LEARNING_RATE = 3e-4
        CLA_LEARNINIG_RATE = 3e-3

        # Training schedule
        EPOCHS = 1000
        TRAIN_SIZE = 60000 - NUM_LABEL  # Num of unlabeled data
        SAVE_PER_EPOCH = 1 # epochs to save trained_weights
        VAL_STEP = None

        # Results
        # SAMPLE_DIR = "cifar10_good_GAN_4000_Unsup"
        SAMPLE_DIR = "cifar10_good_GAN_4000"
        WEIGHT_DIR = os.path.join(root_dir, "Training/Weight_cifar10")
        LOG_DIR = os.path.join(root_dir, "Training/Log_cifar10")

    # Create the global configuration
    tmp_config = TempConfig()
    if FLAGS:
        _customize_config(tmp_config, FLAGS)
    tmp_config.SAMPLE_DIR = os.path.join(root_dir, "Training", tmp_config.SAMPLE_DIR)
    if tmp_config.NUM_LABEL < 1000:
        tmp_config.PRE_TRAIN = True
    tmp_config.display()
    check_folder_exists(tmp_config.SAMPLE_DIR)

    # Comments log on the current run
    comments = "This training is for cifar10 dataset."
    comments += tmp_config.config_str() + datetime.now(timezone('America/Los_Angeles')).strftime("%Y-%m-%d_%H_%M_%S")
    # Create a training object
    training = Train(tmp_config, tmp_config.LOG_DIR, tmp_config.WEIGHT_DIR, comments=comments)
    # Load in sample_y
    sample_y = np.load(os.path.join(root_dir, "DataSet/cifar_10/cifar10_sample_y.npy"))
    training.train(Dataset, Model, sample_y)

def _main_training_mnist(FLAGS = None):
    from config import Config
    from Input_Pipeline.mnistDataset import mnistDataset as Dataset
    from Model.Good_GAN import Good_GAN as Model

    # if FLAGS is not None and FLAGS.simple:
    #     from Model.Good_GAN_simple import Good_GAN_simple as Model
    # else:
    #     from Model.Good_GAN import Good_GAN as Model

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class TempConfig(Config):
        NAME = "Good_GAN"
        ## Input pipeline
        DATA_NAME = "mnist"
        DATA_DIR = os.path.join(root_dir, "DataSet/mnist")
        NUM_LABEL = 100
        BATCH_SIZE_G = 100
        BATCH_SIZE_bG = 100
        BATCH_SIZE_L_C = 100
        BATCH_SIZE_U_C = 100
        BATCH_SIZE_L_D = 20
        BATCH_SIZE_U_D = 80
        BATCH_SIZE = BATCH_SIZE_G

        IMAGE_HEIGHT = 28
        IMAGE_WIDTH = 28
        CHANNEL = 1
        REPEAT = -1

        FAKE_G_LAMBDA = 0.1

        ## Model architecture
        # Number of classification classes
        Z_DIM = 100
        NUM_CLASSES = 10
        MINIBATCH_DIS = False

        ## Traning settings
        # Restore
        RESTORE = True  # Whether to use the previous trained weights
        RUN = "Run_2019-05-20_19_41_48"
        # RESTORE_EPOCH = 30
        LEARNING_RATE = 1e-3

        # Training schedule
        EPOCHS = 1000
        TRAIN_SIZE = 60000 - NUM_LABEL  # Num of unlabeled data
        SAVE_PER_EPOCH = 1 # epochs to save trained_weights
        VAL_STEP = None
        CLA_LEARNINIG_RATE = 3e-4

        # Results
        SAMPLE_DIR = "good_GAN_100_mnist"
        WEIGHT_DIR = os.path.join(root_dir, "Training/Weight_mnist")
        LOG_DIR = os.path.join(root_dir, "Training/Log_mnist")

    # Create the global configuration
    tmp_config = TempConfig()
    if FLAGS:
        _customize_config(tmp_config, FLAGS)
    tmp_config.SAMPLE_DIR = os.path.join(root_dir, "Training", tmp_config.SAMPLE_DIR)
    # if tmp_config.NUM_LABEL < 1000:
    #     tmp_config.PRE_TRAIN = True
    tmp_config.display()
    check_folder_exists(tmp_config.SAMPLE_DIR)

    # Comments log on the current run
    comments = "This training is for mnist dataset."
    comments += tmp_config.config_str() + datetime.now(timezone('America/Los_Angeles')).strftime("%Y-%m-%d_%H_%M_%S")
    # Create a training object
    training = Train(tmp_config, tmp_config.LOG_DIR, tmp_config.WEIGHT_DIR, comments=comments)
    # Load in sample_y
    sample_y = np.load(os.path.join(root_dir, "DataSet/mnist/mnist_sample_y.npy"))
    training.train(Dataset, Model, sample_y)


def _customize_config(tmp_config, FLAGS):
    tmp_config.NAME = FLAGS.name
    tmp_config.EPOCHS = FLAGS.epoch
    tmp_config.LEARNING_RATE = FLAGS.learning_rate
    tmp_config.BETA1 = FLAGS.beta1
    tmp_config.BATCH_SIZE = FLAGS.batch_size
    tmp_config.RESTORE = FLAGS.restore
    if FLAGS.run:
        tmp_config.RUN = os.path.join(tmp_config.WEIGHT_DIR, FLAGS.run)
        tmp_config.RESTORE_EPOCH = FLAGS.restore_epoch
    tmp_config.SAMPLE_DIR = os.path.join(root_dir, "Training", FLAGS.sample_dir)
    tmp_config.MINIBATCH_DIS = FLAGS.miniBatchDis
    tmp_config.DEBUG = FLAGS.debug
    tmp_config.NUM_LABEL = FLAGS.num_label

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    _main_training_mnist()