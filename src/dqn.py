import tensorflow as tf
#import tensorlayer as tl
import numpy as np

class DQN(object):
    def __init__(self, hps, name_variable):
        self._hps = hps
        self._name_variable = name_variable

    def variable_summaries(self, var_name, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries_{}'.format(var_name)):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        self._x = tf.placeholder(tf.float32, [None, self._hps.dqn_input_feature_len], name='x') # size (dataset_len, input_feature_len)
        self._y = tf.placeholder(tf.float32, [None, self._hps.vocab_size], name='y') # size (dataset_len, 1)
        self._train_step = tf.placeholder(tf.int32, None,name='train_step')

    def _make_feed_dict(self, batch):
        feed_dict = {}
        feed_dict[self._x] = batch._x
        feed_dict[self._y] = batch._y
        return feed_dict

    def _add_tf_layers(self):
        """ Based on the dqn_layers flag, it creates multiple dense layers to do the regression. """

        h = tf.layers.dense(self._x, units = self._hps.dqn_input_feature_len, activation=tf.nn.relu, name='{}_input_layer'.format(self._name_variable))
        for i, layer in enumerate(self._hps.dqn_layers.split(',')):
            h = tf.layers.dense(h, units = int(layer), activation = tf.nn.relu, name='{}_h_{}'.format(self._name_variable, i))

        self.advantage_layer = tf.layers.dense(h, units = self._hps.vocab_size, activation = tf.nn.softmax, name='{}_advantage'.format(self._name_variable))
        if self._hps.dueling_net:
            # in dueling net, we have two extra output layers; one for value function estimation
            # and the other for advantage estimation, we then use the difference between these two layers
            # to calculate the q-estimation
            self_layer = tf.layers.dense(h, units = 1, activation = tf.identity, name='{}_value'.format(self._name_variable))
            normalized_al = self.advantage_layer-tf.reshape(tf.reduce_mean(self.advantage_layer,axis=1),[-1,1]) # equation 9 in https://arxiv.org/pdf/1511.06581.pdf
            value_extended = tf.concat([self_layer] * self._hps.vocab_size, axis=1)
            self.output = value_extended + normalized_al
        else:
            self.output = self.advantage_layer

    def _add_train_op(self):
        # In regression, the objective loss is Mean Squared Error (MSE).
        self.loss = tf.losses.mean_squared_error(labels = self._y, predictions = self.output)

        tvars = tf.trainable_variables()
        gradients = tf.gradients(self.loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip the gradients
        with tf.device("/gpu:{}".format(self._hps.dqn_gpu_num)):
            grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

        # Add a summary
        tf.summary.scalar('global_norm', global_norm)

        # Apply adagrad optimizer
        optimizer = tf.train.AdamOptimizer(self._hps.lr)
        with tf.device("/gpu:{}".format(self._hps.dqn_gpu_num)):
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

        self.variable_summaries('dqn_loss',self.loss)

    def _add_update_weights_op(self):
        """ Updates the weight of the target network based on the current network. """
        self.model_trainables = tf.trainable_variables(scope='{}_relay_network'.format(self._name_variable)) # target variables
        self._new_trainables = [tf.placeholder(tf.float32, None,name='trainables_{}'.format(i)) for i in range(len(self.model_trainables))]
        self.assign_ops = []
        if self._hps.dqn_polyak_averaging: # target parameters are slowly updating using: \phi_target = \tau * \phi_target + (1-\tau) * \phi_target
            tau = (tf.cast(self._train_step,tf.float32) % self._hps.dqn_target_update)/float(self._hps.dqn_target_update)
            for i, mt in enumerate(self.model_trainables):
                nt = self._new_trainables[i]
                self.assign_ops.append(mt.assign(tau * mt + (1-tau) * nt))
        else:
          if self._train_step % self._hps.dqn_target_update == 0:
            for i, mt in enumerate(self.model_trainables):
                nt = self._new_trainables[i]
                self.assign_ops.append(mt.assign(nt))

    def build_graph(self):
        with tf.variable_scope('{}_relay_network'.format(self._name_variable)), tf.device("/gpu:{}".format(self._hps.dqn_gpu_num)):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self._add_placeholders()
            self._add_tf_layers()
            self._add_train_op()
            self._add_update_weights_op()
            self._summaries = tf.summary.merge_all()

    def run_train_steps(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {'train_op': self.train_op,
        'summaries': self._summaries,
        'loss': self.loss,
        'global_step': self.global_step}
        return sess.run(to_return, feed_dict)

    def run_test_steps(self, sess, x, y=None, return_loss=False, return_best_action=False):
        # when return_loss is True, the model will return the loss of the prediction
        # return_loss should be False, during estimation (decoding)
        feed_dict = {self._x:x}
        to_return = {'estimates': self.output}
        if return_loss:
            feed_dict.update({self._y:y})
            to_return.update({'loss': self.loss})
        output = sess.run(to_return, feed_dict)
        if return_best_action:
            output['best_action']=np.argmax(output['estimates'],axis=1)

        return output

    def run_update_weights(self, sess, train_step, weights):
        feed_dict = {self._train_step:train_step}
        for i, w in enumerate(weights):
            feed_dict.update({self._new_trainables[i]:w})
        _ = sess.run(self.assign_ops, feed_dict)
