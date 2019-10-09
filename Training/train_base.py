'''
This is a training base function upon which the Train.py was built.
'''
import tensorflow as tf
import sys, os
if os.getcwd().endswith("Path_Semi_GAN"):
    root_dir = os.getcwd()
else:
    root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import utils


class Train_base(object):
    def __init__(self):
        pass

    def _input_fn(self):
        raise NotImplementedError(
            'metirc() is implemented in Model sub classes')

    def _build_train_graph(self):
        raise NotImplementedError(
            'loss() is implemented in Model sub classes')

    def _loss(self, target, network_output):
        raise NotImplementedError(
            'loss() is implemented in Model sub classes')

    def _loss_weight_l2(self, var_list, eta=0.001):
        num_weights = utils.get_trainable_weight_num(var_list)
        loss = tf.add_n([tf.nn.l2_loss(v) for v in var_list]) * eta / num_weights
        return loss

    def _huber_loss(self, labels, predictions, delta=1.0, name="huber_loss"):
        with tf.name_scope(name):
            residual = tf.abs(predictions - labels)
            condition = tf.less(residual, delta)
            small_res = 0.5 * tf.square(residual)
            large_res = delta * residual - 0.5 * tf.square(delta)
        return tf.where(condition, small_res, large_res)

    def _entropy(self, logits):
        with tf.name_scope('Entropy'):
            probs = tf.nn.softmax(logits)
            ent = tf.reduce_mean(- tf.reduce_sum(probs * logits, axis=1, keepdims=True) \
                                 + tf.reduce_logsumexp(logits, axis=1, keepdims=True))
        return ent

    def _balance_entropy(self, logits):
        with tf.name_scope("Balance_Entropy"):
            probs = tf.reduce_mean(tf.nn.softmax(logits), axis=0)
            try:
                ent = -tf.reduce_sum(1.0 / tf.cast(tf.shape(logits)[1], tf.float32) * tf.math.log(probs + 1e-12))
            except:
                ent = -tf.reduce_sum(1.0 / tf.cast(tf.shape(logits)[1], tf.float32) * tf.log(probs + 1e-12))
        return ent


    def _metric(self, labels, network_output):
        raise NotImplementedError(
            'metirc() is implemented in Model sub classes')

    def _train_op(self, optimizer, loss, var_list=None):
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step(),
                                      var_list=var_list)
        return train_op

    def _train_op_w_grads(self, optimizer, loss, var_list=None):
        grads = optimizer.compute_gradients(loss, var_list=var_list)
        train_op = optimizer.apply_gradients(grads)
        return train_op, grads

    def _softmax_cross_entropy_loss_w_logits(self, labels, logits):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, \
                                                          logits=logits)
        loss = tf.reduce_mean(loss)
        return loss

    def _sigmoid_cross_entopy_w_logits(self, labels, logits):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)
        return loss

    def _SGD_w_Momentum_optimizer(self, lr, momentum):
        optimizer = tf.train.MomentumOptimizer(learning_rate = lr,
                                               momentum = momentum)
        return optimizer

    def _Adam_optimizer(self, lr, beta1, name='Adam_optimizer'):
        optimizer = tf.train.AdamOptimizer(
            learning_rate = lr,
            beta1 = beta1,
            name=name
        )
        return optimizer

    def _RMSProp_optimizer(self, lr, name='RMSProp_optimizer'):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate = lr,
            decay=0.9,
            name=name
        )
        return optimizer

    def _accuracy_metric(self, labels, predictions):
        return tf.metrics.accuracy(labels, predictions)

    def _auc_metric(self, labels, predictions, curve='ROC'):
        return tf.metrics.auc(labels, predictions, curve=curve)

    def _loss_GAN(self, D, C, Y, Lambda):
        with tf.name_scope('Loss'):
            D_real, D_real_logits,  D_fake, D_fake_logits, D_unl, D_unl_logits = D
            if self.config.DATA_NAME == "cifar10":
                C_real_logits, C_unl_logits, C_unl_d_logits, C_fake_logits, C_unl_logits_rep = C
                c_loss_unsup = tf.losses.mean_squared_error(C_unl_logits, C_unl_logits_rep)
            else:
                C_real_logits, C_unl_logits, C_unl_d_logits, C_fake_logits = C
            y_g, y_l_c = Y

            d_loss_real = self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_real_logits), logits = D_real_logits)
            d_loss_fake = self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_fake_logits), logits = D_fake_logits)
            d_loss_unl = self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_unl_logits), logits = D_unl_logits)
            d_loss = d_loss_real + 0.5 * d_loss_fake + 0.5 * d_loss_unl

            g_loss = 1 / 2 * self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_fake_logits), logits = D_fake_logits)

            c_loss_real = self._softmax_cross_entropy_loss_w_logits(labels = y_l_c, logits = C_real_logits)
            c_loss_fake = self._softmax_cross_entropy_loss_w_logits(labels = y_g, logits = C_fake_logits)

            ## TODO: think about the c_loss for the unlabeled data
            probs = tf.nn.softmax(C_unl_logits)
            p = tf.reduce_max(probs, axis = 1)
            c_loss_unl = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_unl_logits),
                                                               logits=D_unl_logits), axis = 1)  ## C fools D
            c_loss_unl = tf.reduce_mean(tf.multiply(p, c_loss_unl))

            ## make prediction on unlabeled data confident
            c_loss_confid = self._entropy(C_unl_logits)
            ## make the prediction on unlabeled data even(balanced)
            c_loss_balance = self._balance_entropy(C_unl_logits)
            # c_loss_real = c_loss_real + 0.3 * c_loss_confid + 1e-3 * c_loss_balance
            c_loss_real = c_loss_real + 1e-6 * c_loss_confid + 1e-3 * c_loss_balance

            lambda_1 = Lambda[0]
            c_loss = 0.01 * 0.5 * c_loss_unl + c_loss_real + lambda_1 * c_loss_fake
            if self.config.DATA_NAME == "cifar10":
                lambda_2 = Lambda[1]
                c_loss_unsup = lambda_2 * c_loss_unsup
                c_loss += c_loss_unsup
            ## if lambda_1 = 0.1 then it's normal, elif lambda_1 = 0, it's c_loss_fast
        return d_loss, g_loss, c_loss

    def _loss_BGAN(self, C, Y, Lambda):
        y_l_c = Y[0]
        C_real_logits, C_unl_logits, C_fake_logits, feat_real, feat_unl, feat_fake = C
        c_loss_real = self._softmax_cross_entropy_loss_w_logits(labels = y_l_c, logits = C_real_logits)

        ## bad GAN true-fake loss
        c_loss_unl = -0.5 * tf.reduce_mean(tf.reduce_logsumexp(C_unl_logits, axis = 1)) + \
                         0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_unl_logits, axis = 1)))
        c_loss_confid = 0.1 * self._entropy(C_unl_logits)
        c_loss_balance = 1e-3 * self._balance_entropy(C_unl_logits)
        c_loss_fake = 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_fake_logits, axis = 1)))

        c_loss = c_loss_real + c_loss_unl + c_loss_fake + c_loss_confid + c_loss_balance
        # c_loss = c_loss_real + c_loss_unl + c_loss_fake

        ## bad-G loss
        fm_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(feat_fake, axis = 0) - tf.reduce_mean(feat_unl, axis = 0)))

        # # entropy term via pull-away term
        feat_norm = feat_fake / tf.norm(feat_fake, ord='euclidean', axis=1, \
                                             keepdims=True)
        cosine = tf.tensordot(feat_norm, feat_norm, axes=[[1], [1]])
        mask = tf.ones(tf.shape(cosine)) - tf.diag(tf.ones(tf.shape(cosine)[0]))
        square = tf.reduce_sum(tf.square(tf.multiply(cosine, mask)))
        divident = tf.cast(tf.shape(cosine)[0] * (tf.shape(cosine)[0] - 1), tf.float32)
        pt_loss = 0.8 * tf.divide(square, divident)
        g_loss = fm_loss + pt_loss
        # g_loss = fm_loss
        return g_loss, c_loss

    def _loss_GoodBadGAN(self, D, C, Y, Lambda):
        with tf.name_scope('Loss'):
            D_real, D_real_logits, D_fake, D_fake_logits, D_unl, D_unl_logits = D
            C_real_logits, C_unl_logits, C_unl_d_logits, C_gG_fake_logits, C_bG_fake_logits, C_real_feat, \
            C_unl_feat, C_bG_fake_feat = C
            y_g, y_l_c = Y

            d_loss_real = self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_real_logits), logits = D_real_logits)
            d_loss_fake = self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_fake_logits), logits = D_fake_logits)
            d_loss_unl = self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_unl_logits), logits = D_unl_logits)
            d_loss = d_loss_real + 0.5 * d_loss_fake + 0.5 * d_loss_unl

            # good generator loss
            gG_loss = 1 / 2 * self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_fake_logits), logits = D_fake_logits)

            # bad generator loss
            fm_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(C_bG_fake_feat, 0) - tf.reduce_mean(C_unl_feat, 0)))
            # entropy term via pull-away term
            feat_norm = C_bG_fake_feat / tf.norm(C_bG_fake_feat, ord='euclidean', axis=1, \
                                            keepdims=True)
            pt_loss = tf.reduce_mean(tf.tensordot(feat_norm, feat_norm, axes=[[1], [1]]))
            bG_loss = fm_loss + 0.8 * pt_loss

            ## c_loss: good GAN part
            c_loss_real = self._softmax_cross_entropy_loss_w_logits(labels = y_l_c, logits = C_real_logits)
            c_loss_gG_fake = self._softmax_cross_entropy_loss_w_logits(labels = y_g, logits = C_gG_fake_logits)

            ## TODO: think about the c_loss for the unlabeled data
            probs = tf.nn.softmax(C_unl_logits)
            p = tf.reduce_max(probs, axis = 1)
            c_loss_unl = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_unl_logits),
                                                               logits=D_unl_logits), axis = 1)  ## C fools D
            c_loss_unl = tf.reduce_mean(tf.multiply(p, c_loss_unl))

            ## make prediction on unlabeled data confident
            c_loss_confid = self._entropy(C_unl_logits)
            ## make the prediction on unlabeled data even(balanced)
            c_loss_balance = self._balance_entropy(C_unl_logits)
            # c_loss_real = c_loss_real + 0.3 * c_loss_confid + 1e-3 * c_loss_balance
            lambda_1 = Lambda[0]
            ## if lambda_1 = 0.03 then it's normal, elif lambda_1 = 0, it's c_loss_fast
            c_loss_gG = c_loss_real + 0.3 * c_loss_confid + 0.01 * 0.5 * c_loss_unl + lambda_1 * c_loss_gG_fake \
                        + 1e-3 * c_loss_balance

            ## c_loss: bad GAN part: true-fake loss
            c_loss_bad_unl = -0.5 * tf.reduce_mean(tf.reduce_logsumexp(C_unl_logits, axis = 1)) + \
                             0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_unl_logits, axis = 1)))
            c_loss_bG_fake = 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_bG_fake_logits, axis = 1)))

            ## total c_loss
            c_loss_bG = c_loss_real + c_loss_bad_unl + c_loss_bG_fake

            c_loss = c_loss_gG + c_loss_bG - c_loss_real

        return d_loss, gG_loss, bG_loss, c_loss

    def _loss_GoodRegGAN(self, D, C, Y, Lambda):
        with tf.name_scope('Loss'):
            D_real, D_real_logits, D_fake, D_fake_logits, D_unl, D_unl_logits = D
            C_real_logits, C_unl_logits, C_unl_d_logits, C_gG_fake_logits, C_bG_fake_logits, C_bG_fake_perturb_logits, C_real_feat, \
            C_unl_feat, C_bG_fake_feat, C_bG_fake_perturb_feat = C
            y_g, y_l_c = Y

            d_loss_real = self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_real_logits), logits = D_real_logits)
            d_loss_fake = 0.5 * self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_fake_logits), logits = D_fake_logits)
            d_loss_unl = 0.5 * self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_unl_logits), logits = D_unl_logits)
            d_loss = d_loss_real + d_loss_fake + d_loss_unl

            # good generator loss
            gG_loss = 1 / 2 * self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_fake_logits), logits = D_fake_logits)

            # bad generator loss
            fm_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(C_bG_fake_feat, 0) - tf.reduce_mean(C_unl_feat, 0)))

            ###### entropy term via pull-away term
            feat_norm = C_bG_fake_feat / tf.norm(C_bG_fake_feat, ord='euclidean', axis=1, \
                                            keepdims=True)
            cosine = tf.tensordot(feat_norm, feat_norm, axes=[[1], [1]])
            mask = tf.ones(tf.shape(cosine)) - tf.diag(tf.ones(tf.shape(cosine)[0]))
            square = tf.reduce_sum(tf.square(tf.multiply(cosine, mask)))
            divident = tf.cast(tf.shape(cosine)[0] * (tf.shape(cosine)[0] - 1), tf.float32)
            pt_loss = 0.8 * tf.divide(square, divident)
            #######
            bG_loss = fm_loss + pt_loss

            ## c_loss: good GAN part
            c_loss_real = self._softmax_cross_entropy_loss_w_logits(labels = y_l_c, logits = C_real_logits)
            c_loss_gG_fake = self._softmax_cross_entropy_loss_w_logits(labels = y_g, logits = C_gG_fake_logits)

            ## TODO: think about the c_loss for the unlabeled data
            probs = tf.nn.softmax(C_unl_logits)
            p = tf.reduce_max(probs, axis = 1)
            c_loss_unl = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_unl_logits),
                                                               logits=D_unl_logits), axis = 1)  ## C fools D
            c_loss_unl = 0.01 * 0.5 * tf.reduce_mean(tf.multiply(p, c_loss_unl))

            ## make prediction on unlabeled data confident
            c_loss_confid = 0.3 * self._entropy(C_unl_logits)
            ## make the prediction on unlabeled data even(balanced)
            c_loss_balance = 1e-3 * self._balance_entropy(C_unl_logits)
            # c_loss_real = c_loss_real + 0.3 * c_loss_confid + 1e-3 * c_loss_balance
            lambda_1, lambda_2, lambda_3 = Lambda
            ## if lambda_1 = 0.03 then it's normal, elif lambda_1 = 0, it's c_loss_fast
            c_loss_gG = c_loss_confid + c_loss_unl + lambda_1 * c_loss_gG_fake \
                        + c_loss_balance

            ## c_loss: bad GAN part: true-fake loss
            c_loss_bad_unl = -0.5 * tf.reduce_mean(tf.reduce_logsumexp(C_unl_logits, axis = 1)) + \
                             0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_unl_logits, axis = 1)))
            c_loss_bG_fake = 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_bG_fake_logits, axis = 1)))
            c_loss_bG_fake_perturb = 1e-3 * tf.reduce_mean(tf.reduce_sum(tf.square(C_bG_fake_perturb_logits - C_bG_fake_logits),axis=1))

            ## total c_loss
            c_loss_bG = c_loss_bad_unl + c_loss_bG_fake + c_loss_bG_fake_perturb

            c_loss = c_loss_real + lambda_2 * c_loss_gG + lambda_3 * c_loss_bG

        return [d_loss, d_loss_real, d_loss_fake, d_loss_unl], gG_loss, bG_loss, [c_loss, c_loss_real, c_loss_gG, \
                c_loss_confid, c_loss_unl, c_loss_balance, c_loss_gG_fake, c_loss_bG, c_loss_bad_unl, c_loss_bG_fake, \
                c_loss_bG_fake_perturb]

    def _loss_GoodRegGAN_cifar10(self, D, C, Y, Lambda):
        with tf.name_scope('Loss'):
            D_real, D_real_logits, D_fake, D_fake_logits, D_unl, D_unl_logits = D
            C_real_logits, C_unl_logits, C_unl_d_logits, C_gG_fake_logits, C_bG_fake_logits, C_bG_fake_perturb_logits, C_real_feat, \
            C_unl_feat, C_bG_fake_feat, C_bG_fake_perturb_feat, C_unl_logits_rep = C
            y_g, y_l_c = Y

            d_loss_real = self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_real_logits), logits = D_real_logits)
            d_loss_fake = 0.5 * self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_fake_logits), logits = D_fake_logits)
            d_loss_unl = 0.5 * self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_unl_logits), logits = D_unl_logits)
            d_loss = d_loss_real + d_loss_fake + d_loss_unl

            # good generator loss
            gG_loss = 1 / 2 * self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_fake_logits), logits = D_fake_logits)

            # bad generator loss
            fm_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(C_bG_fake_feat, 0) - tf.reduce_mean(C_unl_feat, 0)))

            ###### entropy term via pull-away term
            feat_norm = C_bG_fake_feat / tf.norm(C_bG_fake_feat, ord='euclidean', axis=1, \
                                            keepdims=True)
            cosine = tf.tensordot(feat_norm, feat_norm, axes=[[1], [1]])
            mask = tf.ones(tf.shape(cosine)) - tf.diag(tf.ones(tf.shape(cosine)[0]))
            square = tf.reduce_sum(tf.square(tf.multiply(cosine, mask)))
            divident = tf.cast(tf.shape(cosine)[0] * (tf.shape(cosine)[0] - 1), tf.float32)
            pt_loss = 0.8 * tf.divide(square, divident)
            #######
            bG_loss = fm_loss + pt_loss

            ## c_loss: good GAN part
            c_loss_real = self._softmax_cross_entropy_loss_w_logits(labels = y_l_c, logits = C_real_logits)
            c_loss_gG_fake = self._softmax_cross_entropy_loss_w_logits(labels = y_g, logits = C_gG_fake_logits)

            ## TODO: think about the c_loss for the unlabeled data
            probs = tf.nn.softmax(C_unl_logits)
            p = tf.reduce_max(probs, axis = 1)
            c_loss_unl = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_unl_logits),
                                                               logits=D_unl_logits), axis = 1)  ## C fools D
            c_loss_unl = 0.01 * 0.5 * tf.reduce_mean(tf.multiply(p, c_loss_unl))

            ## make prediction on unlabeled data confident
            c_loss_confid = 0.3 * self._entropy(C_unl_logits)
            ## make the prediction on unlabeled data even(balanced)
            c_loss_balance = 1e-3 * self._balance_entropy(C_unl_logits)
            # c_loss_real = c_loss_real + 0.3 * c_loss_confid + 1e-3 * c_loss_balance
            lambda_1, lambda_2, lambda_3, lambda_4 = Lambda
            ## if lambda_1 = 0.03 then it's normal, elif lambda_1 = 0, it's c_loss_fast
            c_loss_gG = c_loss_confid + c_loss_unl + lambda_1 * c_loss_gG_fake \
                        + c_loss_balance

            ## c_loss: bad GAN part: true-fake loss
            c_loss_bad_unl = -0.5 * tf.reduce_mean(tf.reduce_logsumexp(C_unl_logits, axis = 1)) + \
                             0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_unl_logits, axis = 1)))
            c_loss_bG_fake = 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_bG_fake_logits, axis = 1)))
            c_loss_bG_fake_perturb = 1e-3 * tf.reduce_mean(tf.reduce_sum(tf.square(C_bG_fake_perturb_logits - C_bG_fake_logits),axis=1))


            ## unsup loss for cifar10:
            c_loss_unsup = lambda_4 * tf.losses.mean_squared_error(C_unl_logits, C_unl_logits_rep)

            ## total c_loss
            c_loss_bG = c_loss_bad_unl + c_loss_bG_fake + c_loss_bG_fake_perturb

            c_loss = c_loss_real + lambda_2 * c_loss_gG + lambda_3 * c_loss_bG + c_loss_unsup

        return [d_loss, d_loss_real, d_loss_fake, d_loss_unl], gG_loss, bG_loss, [c_loss, c_loss_real, c_loss_gG, \
                c_loss_confid, c_loss_unl, c_loss_balance, c_loss_gG_fake, c_loss_bG, c_loss_bad_unl, c_loss_bG_fake, \
                c_loss_bG_fake_perturb, c_loss_unsup]

    def _loss_GoodRegGAN_BS(self, D, C, Y, Lambda):
        with tf.name_scope('Loss'):
            D_real, D_real_logits, D_fake, D_fake_logits, D_unl, D_unl_logits = D
            C_real_logits, C_unl_logits, C_unl_d_logits, C_gG_fake_logits, C_bG_fake_logits, C_bG_fake_perturb_logits, C_unl_logits_bG, C_real_feat, \
            C_unl_feat, C_bG_fake_feat, C_bG_fake_perturb_feat, C_unl_feat_bG = C
            y_g, y_l_c = Y

            d_loss_real = self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_real_logits), logits = D_real_logits)
            d_loss_fake = 0.5 * self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_fake_logits), logits = D_fake_logits)
            d_loss_unl = 0.5 * self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_unl_logits), logits = D_unl_logits)
            d_loss = d_loss_real + d_loss_fake + d_loss_unl

            # good generator loss
            gG_loss = 0.5 * self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_fake_logits), logits = D_fake_logits)

            # bad generator loss
            fm_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(C_bG_fake_feat, 0) - tf.reduce_mean(C_unl_feat_bG, 0)))

            # # entropy term via pull-away term
            # feat_norm = C_bG_fake_feat / tf.norm(C_bG_fake_feat, ord='euclidean', axis=1, \
            #                                      keepdims=True)
            # cosine = tf.tensordot(feat_norm, feat_norm, axes=[[1], [1]])
            # mask = tf.ones(tf.shape(cosine)) - tf.diag(tf.ones(tf.shape(cosine)[0]))
            # square = tf.reduce_sum(tf.square(tf.multiply(cosine, mask)))
            # divident = tf.cast(tf.shape(cosine)[0] * (tf.shape(cosine)[0] - 1), tf.float32)
            # pt_loss = 0.8 * tf.divide(square, divident)
            #######

            # bG_loss = fm_loss + 0.8 * pt_loss
            bG_loss = fm_loss

            ## c_loss: supervised signal
            c_loss_real = self._softmax_cross_entropy_loss_w_logits(labels = y_l_c, logits = C_real_logits)

            ## c_loss: good GAN part
            c_loss_gG_fake = self._softmax_cross_entropy_loss_w_logits(labels = y_g, logits = C_gG_fake_logits)

            probs = tf.nn.softmax(C_unl_logits)
            p = tf.reduce_max(probs, axis = 1)
            c_loss_unl = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_unl_logits),
                                                               logits=D_unl_logits), axis = 1)  ## C fools D
            c_loss_unl = 0.01 * 0.5 * tf.reduce_mean(tf.multiply(p, c_loss_unl))

            ## make prediction on unlabeled data confident
            c_loss_confid = 1e-5 * self._entropy(C_unl_logits)
            ## make the prediction on unlabeled data even(balanced)
            c_loss_balance = 1e-3 * self._balance_entropy(C_unl_logits)

            lambda_1, lambda_2, lambda_3 = Lambda
            ## if lambda_1 = 0.03 then it's normal, elif lambda_1 = 0, it's c_loss_fast
            c_loss_gG =  c_loss_confid + c_loss_unl + lambda_1 * c_loss_gG_fake \
                        + c_loss_balance

            ## c_loss: bad GAN part: true-fake loss
            c_loss_bad_unl = -0.5 * tf.reduce_mean(tf.reduce_logsumexp(C_unl_logits, axis = 1)) + \
                             0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_unl_logits, axis = 1)))
            c_loss_bG_fake = 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_bG_fake_logits, axis = 1)))
            c_loss_bG_fake_perturb = 1e-3 * tf.reduce_mean(tf.reduce_sum(tf.square(C_bG_fake_perturb_logits - C_bG_fake_logits),axis=1))

            ## total c_loss
            c_loss_bG = c_loss_bad_unl + c_loss_bG_fake + c_loss_bG_fake_perturb

            c_loss = c_loss_real + lambda_2 * c_loss_gG + lambda_3 * c_loss_bG

        return [d_loss, d_loss_real, d_loss_fake, d_loss_unl], gG_loss, bG_loss, [c_loss, c_loss_real, c_loss_gG, \
                c_loss_confid, c_loss_unl, c_loss_balance, c_loss_gG_fake, c_loss_bG, c_loss_bad_unl, c_loss_bG_fake, \
                c_loss_bG_fake_perturb]

    def _loss_GoodRegGAN_BS_cifar10(self, D, C, Y, Lambda):
        with tf.name_scope('Loss'):
            D_real, D_real_logits, D_fake, D_fake_logits, D_unl, D_unl_logits = D
            C_real_logits, C_unl_logits, C_unl_d_logits, C_gG_fake_logits, C_bG_fake_logits, C_bG_fake_perturb_logits, C_unl_logits_bG, C_real_feat, \
            C_unl_feat, C_bG_fake_feat, C_bG_fake_perturb_feat, C_unl_feat_bG, C_unl_logits_rep = C
            y_g, y_l_c = Y

            d_loss_real = self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_real_logits), logits = D_real_logits)
            d_loss_fake = 0.5 * self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_fake_logits), logits = D_fake_logits)
            d_loss_unl = 0.5 * self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_unl_logits), logits = D_unl_logits)
            d_loss = d_loss_real + d_loss_fake + d_loss_unl

            # good generator loss
            gG_loss = 0.5 * self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_fake_logits), logits = D_fake_logits)

            # bad generator loss
            fm_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(C_bG_fake_feat, 0) - tf.reduce_mean(C_unl_feat_bG, 0)))

            # # entropy term via pull-away term
            # feat_norm = C_bG_fake_feat / tf.norm(C_bG_fake_feat, ord='euclidean', axis=1, \
            #                                      keepdims=True)
            # cosine = tf.tensordot(feat_norm, feat_norm, axes=[[1], [1]])
            # mask = tf.ones(tf.shape(cosine)) - tf.diag(tf.ones(tf.shape(cosine)[0]))
            # square = tf.reduce_sum(tf.square(tf.multiply(cosine, mask)))
            # divident = tf.cast(tf.shape(cosine)[0] * (tf.shape(cosine)[0] - 1), tf.float32)
            # pt_loss = 0.8 * tf.divide(square, divident)
            # pt_loss = 0
            # bG_loss = fm_loss + pt_loss
            bG_loss = fm_loss

            ## c_loss: supervised signal
            c_loss_real = self._softmax_cross_entropy_loss_w_logits(labels = y_l_c, logits = C_real_logits)

            ## c_loss: good GAN part
            if self.config.FAST_MODE:
                c_loss_gG_fake = 0
            else:
                c_loss_gG_fake = self._softmax_cross_entropy_loss_w_logits(labels = y_g, logits = C_gG_fake_logits)

            probs = tf.nn.softmax(C_unl_logits)
            p = tf.reduce_max(probs, axis = 1)
            c_loss_unl = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_unl_logits),
                                                               logits=D_unl_logits), axis = 1)  ## C fools D
            c_loss_unl = 1e-2 * 0.5 * tf.reduce_mean(tf.multiply(p, c_loss_unl))

            ## make prediction on unlabeled data confident
            c_loss_confid = 1e-7 * self._entropy(C_unl_logits)
            ## make the prediction on unlabeled data even(balanced)
            c_loss_balance = 1e-3 * self._balance_entropy(C_unl_logits)

            lambda_1, lambda_2, lambda_3, lambda_4 = Lambda
            ## if lambda_1 = 0.03 then it's normal, elif lambda_1 = 0, it's c_loss_fast
            c_loss_gG =  c_loss_confid + c_loss_unl + lambda_1 * c_loss_gG_fake \
                        + c_loss_balance

            ## c_loss: bad GAN part: true-fake loss
            c_loss_bad_unl = -0.5 * tf.reduce_mean(tf.reduce_logsumexp(C_unl_logits, axis = 1)) + \
                             0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_unl_logits, axis = 1)))
            c_loss_bG_fake = 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_bG_fake_logits, axis = 1)))
            c_loss_bG_fake_perturb = 1e-3 * tf.reduce_mean(tf.reduce_sum(tf.square(C_bG_fake_perturb_logits - C_bG_fake_logits),axis=1))

            ## unsup loss for cifar10:
            # c_loss_unsup = lambda_4 * tf.losses.mean_squared_error(C_unl_logits, C_unl_logits_rep)
            c_loss_unsup = lambda_4
            ## total c_loss
            c_loss_bG = c_loss_bad_unl + c_loss_bG_fake + c_loss_bG_fake_perturb

            c_loss = c_loss_real + lambda_2 * c_loss_gG + lambda_3 * c_loss_bG + c_loss_unsup

        return [d_loss, d_loss_real, d_loss_fake, d_loss_unl], gG_loss, bG_loss, [c_loss, c_loss_real, c_loss_gG, \
                c_loss_confid, c_loss_unl, c_loss_balance, c_loss_gG_fake, c_loss_bG, c_loss_bad_unl, c_loss_bG_fake, \
                c_loss_bG_fake_perturb, c_loss_unsup]



    def _loss_GoodRegBadGAN(self, D, C, Y, Lambda):
        with tf.name_scope('Loss'):
            D_real, D_real_logits, D_fake, D_fake_logits, D_unl, D_unl_logits = D
            C_real_logits, C_unl_logits, C_unl_d_logits, C_gG_fake_logits, C_bG_fake_logits, C_bG_fake_perturb_logits, C_real_feat, \
            C_unl_feat, C_bG_fake_feat, C_bG_fake_perturb_feat = C
            y_g, y_l_c = Y

            d_loss_real = self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_real_logits), logits = D_real_logits)
            d_loss_fake = self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_fake_logits), logits = D_fake_logits)
            d_loss_unl = self._sigmoid_cross_entopy_w_logits(labels = tf.zeros_like(D_unl_logits), logits = D_unl_logits)
            d_loss = d_loss_real + 0.5 * d_loss_fake + 0.5 * d_loss_unl

            # good generator loss
            gG_loss = 1 / 2 * self._sigmoid_cross_entopy_w_logits(labels = tf.ones_like(D_fake_logits), logits = D_fake_logits)

            # bad generator loss
            fm_loss = tf.reduce_mean(tf.abs(tf.reduce_mean(C_bG_fake_feat, 0) - tf.reduce_mean(C_unl_feat, 0)))
            # entropy term via pull-away term
            feat_norm = C_bG_fake_feat / tf.norm(C_bG_fake_feat, ord='euclidean', axis=1, \
                                            keepdims=True)
            pt_loss = tf.reduce_mean(tf.tensordot(feat_norm, feat_norm, axes=[[1], [1]]))
            bG_loss = fm_loss + 0.8 * pt_loss

            ## c_loss: good GAN part
            c_loss_real = self._softmax_cross_entropy_loss_w_logits(labels = y_l_c, logits = C_real_logits)
            c_loss_gG_fake = self._softmax_cross_entropy_loss_w_logits(labels = y_g, logits = C_gG_fake_logits)

            ## TODO: think about the c_loss for the unlabeled data
            probs = tf.nn.softmax(C_unl_logits)
            p = tf.reduce_max(probs, axis = 1)
            c_loss_unl = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_unl_logits),
                                                               logits=D_unl_logits), axis = 1)  ## C fools D
            c_loss_unl = tf.reduce_mean(tf.multiply(p, c_loss_unl))

            ## make prediction on unlabeled data confident
            c_loss_confid = self._entropy(C_unl_logits)
            ## make the prediction on unlabeled data even(balanced)
            c_loss_balance = self._balance_entropy(C_unl_logits)
            # c_loss_real = c_loss_real + 0.3 * c_loss_confid + 1e-3 * c_loss_balance
            lambda_1 = Lambda[0]
            ## if lambda_1 = 0.03 then it's normal, elif lambda_1 = 0, it's c_loss_fast
            c_loss_gG = c_loss_real + 0.3 * c_loss_confid + 0.01 * 0.5 * c_loss_unl + lambda_1 * c_loss_gG_fake \
                        + 1e-3 * c_loss_balance

            ## c_loss: bad GAN part: true-fake loss
            c_loss_bad_unl = -0.5 * tf.reduce_mean(tf.reduce_logsumexp(C_unl_logits, axis = 1)) + \
                             0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_unl_logits, axis = 1)))
            c_loss_bG_fake = 0.5 * tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(C_bG_fake_logits, axis = 1)))
            c_loss_bG_fake_perturb = 1e-3 * tf.reduce_mean(tf.reduce_sum(tf.square(C_bG_fake_perturb_logits - C_bG_fake_logits),axis=1))

            ## total c_loss
            c_loss_bG = c_loss_real + c_loss_bad_unl + c_loss_bG_fake + c_loss_bG_fake_perturb

            c_loss = c_loss_gG + c_loss_bG - c_loss_real

        return d_loss, gG_loss, bG_loss, c_loss

    def _loss_WGAN_GP(self, G, D, C, X, Y, Lambda, discriminator):
        with tf.name_scope('Loss'):
            D_real, D_real_logits, fm_real, D_fake, D_fake_logits, fm_fake, D_unlabel, D_unlabel_logits, fm_unlabel = D
            C_real_logits, C_fake_logits, C_unlabel_logits = C
            lambda_1, lambda_2 = Lambda

            wd1 = 1 / 2 * (tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_fake_logits))
            wd2 = 1 / 2 * (tf.reduce_mean(D_real_logits) - tf.reduce_mean(D_unlabel_logits))
            wd3 = 1 / 2 * (tf.reduce_mean(D_unlabel_logits) - tf.reduce_mean(D_fake_logits))
            gp = self._gradient_penalty(X, G, Y, discriminator)
            d_loss = -(wd1 + lambda_1 * wd2 + lambda_2 * wd3) + gp * 10.0

            g_loss = -tf.reduce_mean(D_fake_logits)

            c_loss_real = self._softmax_cross_entropy_loss_w_logits(labels = Y, logits = C_real_logits)
            c_loss_fake = self._softmax_cross_entropy_loss_w_logits(labels = Y, logits = C_fake_logits)
            ## TODO: think about the c_loss for the unlabeled data
            # max_c = tf.cast(tf.argmax(C_unlabel_logits, axis = 1), tf.float32)
            # c_loss_unlabel = tf.reduce_mean(max_c * tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(D_unlabel_logits),\
            #                                                                                 logits = D_unlabel_logits))

            # c_loss = c_loss_real + lambda_fake * c_loss_fake + 1/2 * 0.01 * c_loss_unlabel
            c_loss = c_loss_real + lambda_2 * c_loss_fake
        return d_loss, g_loss, c_loss

    def _gradient_penalty(self, real, fake, label, f):
        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        _, pred, _ = f(x, y=label, reuse=True)
        gradients = tf.gradients(pred, x)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp