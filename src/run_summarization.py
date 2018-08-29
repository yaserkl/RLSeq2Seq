# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This is the top-level file to train, evaluate or test your summarization model"""

import time
import os
import tensorflow as tf
from collections import namedtuple
from data import Vocab
from batcher import Batcher
from model import SummarizationModel
from decode import BeamSearchDecoder
import util as util
import numpy as np
from glob import glob
from tensorflow.python import debug as tf_debug
from replay_buffer import ReplayBuffer
from dqn import DQN
from threading import Thread
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bernoulli

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_integer('decode_after', 0, 'skip already decoded docs')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# batcher parameter, for consistent results, set all these parameters to 1
tf.app.flags.DEFINE_integer('example_queue_threads', 4, 'Number of example queue threads,')
tf.app.flags.DEFINE_integer('batch_queue_threads', 2, 'Number of batch queue threads.')
tf.app.flags.DEFINE_integer('bucketing_cache_size', 100, 'Number of bucketing cache size.')

# Hyperparameters
tf.app.flags.DEFINE_integer('enc_hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('dec_hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 64, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('max_iter', 55000, 'max number of iterations')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
tf.app.flags.DEFINE_string('embedding', None, 'path to the pre-trained embedding file')
tf.app.flags.DEFINE_integer('gpu_num', 0, 'which gpu to use to train the model')

# Pointer-generator or baseline model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Pointer-generator with Self-Critic policy gradient: https://arxiv.org/pdf/1705.04304.pdf
tf.app.flags.DEFINE_boolean('rl_training', False, 'Use policy-gradient training by collecting rewards at the end of sequence.')
tf.app.flags.DEFINE_boolean('self_critic', True, 'Uses greedy sentence reward as baseline.')
tf.app.flags.DEFINE_boolean('use_discounted_rewards', False, 'Whether to use discounted rewards.')
tf.app.flags.DEFINE_boolean('use_intermediate_rewards', False, 'Whether to use intermediate rewards.')
tf.app.flags.DEFINE_boolean('convert_to_reinforce_model', False, 'Convert a pointer model to a reinforce model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('intradecoder', False, 'Use intradecoder attention or not')
tf.app.flags.DEFINE_boolean('use_temporal_attention', False, 'Whether to use temporal attention or not')
tf.app.flags.DEFINE_boolean('matrix_attention', False, 'Use matrix attention, Eq. 2 https://arxiv.org/pdf/1705.04304.pdf')
tf.app.flags.DEFINE_float('eta', 0, 'RL/MLE scaling factor, 1 means use RL loss, 0 means use MLE loss')
tf.app.flags.DEFINE_boolean('fixed_eta', False, 'Use fixed value for eta or adaptive based on global step')
tf.app.flags.DEFINE_float('gamma', 0.99, 'discount factor')
tf.app.flags.DEFINE_string('reward_function', 'rouge_l/f_score', 'either bleu or one of the rouge measures (rouge_1/f_score,rouge_2/f_score,rouge_l/f_score)')

# parameters of DDQN model
tf.app.flags.DEFINE_boolean('ac_training', False, 'Use Actor-Critic learning by DDQN.')
tf.app.flags.DEFINE_boolean('dqn_scheduled_sampling', False, 'Whether to use scheduled sampling to use estimates of dqn model vs the actual q-estimates values')
tf.app.flags.DEFINE_string('dqn_layers', '512,256,128', 'DQN dense hidden layer size, will create three dense layers with 512, 256, and 128 size')
tf.app.flags.DEFINE_integer('dqn_replay_buffer_size', 100000, 'Size of the replay buffer')
tf.app.flags.DEFINE_integer('dqn_batch_size', 100, 'Batch size for training the DDQN model')
tf.app.flags.DEFINE_integer('dqn_target_update', 10000, 'Update target Q network every 10000 steps')
tf.app.flags.DEFINE_integer('dqn_sleep_time', 2, 'Train DDQN model every 2 seconds')
tf.app.flags.DEFINE_integer('dqn_gpu_num', 0, 'GPU number to train the DDQN')
tf.app.flags.DEFINE_boolean('dueling_net', True, 'Whether to use Duelling Network to train the model') # https://arxiv.org/pdf/1511.06581.pdf
tf.app.flags.DEFINE_boolean('dqn_polyak_averaging', True, 'Whether to use polyak averaging to update the target network parameters')
tf.app.flags.DEFINE_boolean('calculate_true_q', False, "Whether to use true Q-values to train DQN or use DQN's estimates to train it")
tf.app.flags.DEFINE_boolean('dqn_pretrain', False, "Pretrain the DDQN network with fixed Actor model")
tf.app.flags.DEFINE_integer('dqn_pretrain_steps', 10000, 'Number of steps to pre-train the DDQN')

#scheduled sampling parameters, https://arxiv.org/pdf/1506.03099.pdf
# At each time step t and for each sequence in the batch, we get the input to next decoding step by either
#   (1) sampling from the final distribution at (t-1), or
#   (2) reading from input_decoder_embedding.
# We do (1) with probability sampling_probability and (2) with 1 - sampling_probability.
# Using sampling_probability=0.0 is equivalent to using only the ground truth data (no sampling).
# Using sampling_probability=1.0 is equivalent to doing inference by only relying on the sampled token generated at each decoding step
tf.app.flags.DEFINE_boolean('scheduled_sampling', False, 'whether to do scheduled sampling or not')
tf.app.flags.DEFINE_string('decay_function', 'linear','linear, exponential, inv_sigmoid') #### TODO: implement this
tf.app.flags.DEFINE_float('sampling_probability', 0, 'epsilon value for choosing ground-truth or model output')
tf.app.flags.DEFINE_boolean('fixed_sampling_probability', False, 'Whether to use fixed sampling probability or adaptive based on global step')
tf.app.flags.DEFINE_boolean('hard_argmax', True, 'Whether to use soft argmax or hard argmax')
tf.app.flags.DEFINE_boolean('greedy_scheduled_sampling', False, 'Whether to use greedy approach or sample for the output, if True it uses greedy')
tf.app.flags.DEFINE_boolean('E2EBackProp', False, 'Whether to use E2EBackProp algorithm to solve exposure bias')
tf.app.flags.DEFINE_float('alpha', 1, 'soft argmax argument')
tf.app.flags.DEFINE_integer('k', 1, 'number of samples')
tf.app.flags.DEFINE_boolean('scheduled_sampling_final_dist', True, 'Whether to use final distribution or vocab distribution for scheduled sampling')


# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")

class Seq2Seq(object):

  def calc_running_avg_loss(self, loss, running_avg_loss, step, decay=0.99):
    """Calculate the running average loss via exponential decay.
    This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

    Args:
      loss: loss on the most recent eval step
      running_avg_loss: running_avg_loss so far
      summary_writer: FileWriter object to write for tensorboard
      step: training iteration step
      decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

    Returns:
      running_avg_loss: new running average loss
    """
    if running_avg_loss == 0:  # on the first iteration just take the loss
      running_avg_loss = loss
    else:
      running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    self.summary_writer.add_summary(loss_sum, step)
    tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss

  def restore_best_model(self):
    """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
    tf.logging.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=util.get_config())
    print("Initializing all variables...")
    sess.run(tf.initialize_all_variables())

    # Restore the best model from eval dir
    saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = util.load_ckpt(saver, sess, "eval")
    print("Restored %s." % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
    print("Saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print("Saved.")
    exit()

  def restore_best_eval_model(self):
    # load best evaluation loss so far
    best_loss = None
    best_step = None
    # goes through all event files and select the best loss achieved and return it
    event_files = sorted(glob('{}/eval/events*'.format(FLAGS.log_root)))
    for ef in event_files:
      try:
        for e in tf.train.summary_iterator(ef):
          for v in e.summary:
            step = e.step
            if 'running_avg_loss/decay' in v.tag:
              running_avg_loss = v.simple_value
              if best_loss is None or running_avg_loss < best_loss:
                best_loss = running_avg_loss
                best_step = step
      except:
        continue
    tf.logging.info('resotring best loss from the current logs: {}\tstep: {}'.format(best_loss, best_step))
    return best_loss

  def convert_to_coverage_model(self):
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    tf.logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=util.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = util.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()

  def convert_to_reinforce_model(self):
    """Load non-reinforce checkpoint, add initialized extra variables for reinforce, and save as new checkpoint"""
    tf.logging.info("converting non-reinforce model to reinforce model..")

    # initialize an entire reinforce model from scratch
    sess = tf.Session(config=util.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-reinforce weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "reinforce" not in v.name and "Adagrad" not in v.name])
    print("restoring non-reinforce variables...")
    curr_ckpt = util.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_rl_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()

  def setup_training(self):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    if FLAGS.ac_training:
      dqn_train_dir = os.path.join(FLAGS.log_root, "dqn", "train")
      if not os.path.exists(dqn_train_dir): os.makedirs(dqn_train_dir)
    #replaybuffer_pcl_path = os.path.join(FLAGS.log_root, "replaybuffer.pcl")
    #if not os.path.exists(dqn_target_train_dir): os.makedirs(dqn_target_train_dir)

    self.model.build_graph() # build the graph

    if FLAGS.convert_to_reinforce_model:
      assert (FLAGS.rl_training or FLAGS.ac_training), "To convert your pointer model to a reinforce model, run with convert_to_reinforce_model=True and either rl_training=True or ac_training=True"
      self.convert_to_reinforce_model()
    if FLAGS.convert_to_coverage_model:
      assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
      self.convert_to_coverage_model()
    if FLAGS.restore_best_model:
      self.restore_best_model()
    saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time

    # Loads pre-trained word-embedding. By default the model learns the embedding.
    if FLAGS.embedding:
      self.vocab.LoadWordEmbedding(FLAGS.embedding, FLAGS.emb_dim)
      word_vector = self.vocab.getWordEmbedding()

    self.sv = tf.train.Supervisor(logdir=train_dir,
                       is_chief=True,
                       saver=saver,
                       summary_op=None,
                       save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                       save_model_secs=60, # checkpoint every 60 secs
                       global_step=self.model.global_step,
                       init_feed_dict= {self.model.embedding_place:word_vector} if FLAGS.embedding else None
                       )
    self.summary_writer = self.sv.summary_writer
    self.sess = self.sv.prepare_or_wait_for_session(config=util.get_config())
    if FLAGS.ac_training:
      tf.logging.info('DDQN building graph')
      t1 = time.time()
      # We create a separate graph for DDQN
      self.dqn_graph = tf.Graph()
      with self.dqn_graph.as_default():
        self.dqn.build_graph() # build dqn graph
        tf.logging.info('building current network took {} seconds'.format(time.time()-t1))

        self.dqn_target.build_graph() # build dqn target graph
        tf.logging.info('building target network took {} seconds'.format(time.time()-t1))

        dqn_saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time
        self.dqn_sv = tf.train.Supervisor(logdir=dqn_train_dir,
                           is_chief=True,
                           saver=dqn_saver,
                           summary_op=None,
                           save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                           save_model_secs=60, # checkpoint every 60 secs
                           global_step=self.dqn.global_step,
                           )
        self.dqn_summary_writer = self.dqn_sv.summary_writer
        self.dqn_sess = self.dqn_sv.prepare_or_wait_for_session(config=util.get_config())
      ''' #### TODO: try loading a previously saved replay buffer
      # right now this doesn't work due to running DQN on a thread
      if os.path.exists(replaybuffer_pcl_path):
        tf.logging.info('Loading Replay Buffer...')
        try:
          self.replay_buffer = pickle.load(open(replaybuffer_pcl_path, "rb"))
          tf.logging.info('Replay Buffer loaded...')
        except:
          tf.logging.info('Couldn\'t load Replay Buffer file...')
          self.replay_buffer = ReplayBuffer(self.dqn_hps)
      else:
        self.replay_buffer = ReplayBuffer(self.dqn_hps)
      tf.logging.info("Building DDQN took {} seconds".format(time.time()-t1))
      '''
      self.replay_buffer = ReplayBuffer(self.dqn_hps)
    tf.logging.info("Preparing or waiting for session...")
    tf.logging.info("Created session.")
    try:
      self.run_training() # this is an infinite loop until interrupted
    except (KeyboardInterrupt, SystemExit):
      tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
      self.sv.stop()
      if FLAGS.ac_training:
        self.dqn_sv.stop()

  def run_training(self):
    """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
    tf.logging.info("Starting run_training")

    if FLAGS.debug: # start the tensorflow debugger
      self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
      self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    self.train_step = 0
    if FLAGS.ac_training:
      # DDQN training is done asynchronously along with model training
      tf.logging.info('Starting DQN training thread...')
      self.dqn_train_step = 0
      self.thrd_dqn_training = Thread(target=self.dqn_training)
      self.thrd_dqn_training.daemon = True
      self.thrd_dqn_training.start()

      watcher = Thread(target=self.watch_threads)
      watcher.daemon = True
      watcher.start()
    # starting the main thread
    tf.logging.info('Starting Seq2Seq training...')
    while True: # repeats until interrupted
      batch = self.batcher.next_batch()
      t0=time.time()
      if FLAGS.ac_training:
        # For DDQN, we first collect the model output to calculate the reward and Q-estimates
        # Then we fix the estimation either using our target network or using the true Q-values
        # This process will usually take time and we are working on improving it.
        transitions = self.model.collect_dqn_transitions(self.sess, batch, self.train_step, batch.max_art_oovs) # len(batch_size * k * max_dec_steps)
        tf.logging.info('Q-values collection time: {}'.format(time.time()-t0))
        # whenever we are working with the DDQN, we switch using DDQN graph rather than default graph
        with self.dqn_graph.as_default():
          batch_len = len(transitions)
          # we use current decoder state to predict q_estimates, use_state_prime = False
          b = ReplayBuffer.create_batch(self.dqn_hps, transitions,len(transitions), use_state_prime = False, max_art_oovs = batch.max_art_oovs)
          # we also get the next decoder state to correct the estimation, use_state_prime = True
          b_prime = ReplayBuffer.create_batch(self.dqn_hps, transitions,len(transitions), use_state_prime = True, max_art_oovs = batch.max_art_oovs)
          # use current DQN to estimate values from current decoder state
          dqn_results = self.dqn.run_test_steps(sess=self.dqn_sess, x= b._x, return_best_action=True)
          q_estimates = dqn_results['estimates'] # shape (len(transitions), vocab_size)
          dqn_best_action = dqn_results['best_action']
          #dqn_q_estimate_loss = dqn_results['loss']

          # use target DQN to estimate values for the next decoder state
          dqn_target_results = self.dqn_target.run_test_steps(self.dqn_sess, x= b_prime._x)
          q_vals_new_t = dqn_target_results['estimates'] # shape (len(transitions), vocab_size)

          # we need to expand the q_estimates to match the input batch max_art_oov
          # we use the q_estimate of UNK token for all the OOV tokens
          q_estimates = np.concatenate([q_estimates,
            np.reshape(q_estimates[:,0],[-1,1])*np.ones((len(transitions),batch.max_art_oovs))],axis=-1)
          # modify Q-estimates using the result collected from current and target DQN.
          # check algorithm 5 in the paper for more info: https://arxiv.org/pdf/1805.09461.pdf
          for i, tr in enumerate(transitions):
            if tr.done:
              q_estimates[i][tr.action] = tr.reward
            else:
              q_estimates[i][tr.action] = tr.reward + FLAGS.gamma * q_vals_new_t[i][dqn_best_action[i]]
          # use scheduled sampling to whether use true Q-values or DDQN estimation
          if FLAGS.dqn_scheduled_sampling:
            q_estimates = self.scheduled_sampling(batch_len, FLAGS.sampling_probability, b._y_extended, q_estimates)
          if not FLAGS.calculate_true_q:
            # when we are not training DDQN based on true Q-values,
            # we need to update Q-values in our transitions based on the q_estimates we collected from DQN current network.
            for trans, q_val in zip(transitions,q_estimates):
              trans.q_values = q_val # each have the size vocab_extended
          q_estimates = np.reshape(q_estimates, [FLAGS.batch_size, FLAGS.k, FLAGS.max_dec_steps, -1]) # shape (batch_size, k, max_dec_steps, vocab_size_extended)
        # Once we are done with modifying Q-values, we can use them to train the DDQN model.
        # In this paper, we use a priority experience buffer which always selects states with higher quality
        # to train the DDQN. The following line will add batch_size * max_dec_steps experiences to the replay buffer.
        # As mentioned before, the DDQN training is asynchronous. Therefore, once the related queues for DDQN training
        # are full, the DDQN will start the training.
        self.replay_buffer.add(transitions)
        # If dqn_pretrain flag is on, it means that we use a fixed Actor to only collect experiences for
        # DDQN pre-training
        if FLAGS.dqn_pretrain:
          tf.logging.info('RUNNNING DQN PRETRAIN: Adding data to relplay buffer only...')
          continue
        # if not, use the q_estimation to update the loss.
        results = self.model.run_train_steps(self.sess, batch, self.train_step, q_estimates)
      else:
          results = self.model.run_train_steps(self.sess, batch, self.train_step)
      t1=time.time()
      # get the summaries and iteration number so we can write summaries to tensorboard
      summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
      self.train_step = results['global_step'] # we need this to update our running average loss
      tf.logging.info('seconds for training step {}: {}'.format(self.train_step, t1-t0))

      printer_helper = {}
      printer_helper['pgen_loss']= results['pgen_loss']
      if FLAGS.coverage:
        printer_helper['coverage_loss'] = results['coverage_loss']
        if FLAGS.rl_training or FLAGS.ac_training:
          printer_helper['rl_cov_total_loss']= results['reinforce_cov_total_loss']
        else:
          printer_helper['pointer_cov_total_loss'] = results['pointer_cov_total_loss']
      if FLAGS.rl_training or FLAGS.ac_training:
        printer_helper['shared_loss'] = results['shared_loss']
        printer_helper['rl_loss'] = results['rl_loss']
        printer_helper['rl_avg_logprobs'] = results['rl_avg_logprobs']
      if FLAGS.rl_training:
        printer_helper['sampled_r'] = np.mean(results['sampled_sentence_r_values'])
        printer_helper['greedy_r'] = np.mean(results['greedy_sentence_r_values'])
        printer_helper['r_diff'] = printer_helper['sampled_r'] - printer_helper['greedy_r']
      if FLAGS.ac_training:
        printer_helper['dqn_loss'] = np.mean(self.avg_dqn_loss) if len(self.avg_dqn_loss)>0 else 0

      for (k,v) in printer_helper.items():
        if not np.isfinite(v):
          raise Exception("{} is not finite. Stopping.".format(k))
        tf.logging.info('{}: {}\t'.format(k,v))
      tf.logging.info('-------------------------------------------')

      self.summary_writer.add_summary(summaries, self.train_step) # write the summaries
      if self.train_step % 100 == 0: # flush the summary writer every so often
        self.summary_writer.flush()
      if FLAGS.ac_training:
        self.dqn_summary_writer.flush()
      if self.train_step > FLAGS.max_iter: break

  def dqn_training(self):
    """ training the DDQN network."""
    try:
      while True:
        if self.dqn_train_step == FLAGS.dqn_pretrain_steps: raise SystemExit()
        _t = time.time()
        self.avg_dqn_loss = []
        avg_dqn_target_loss = []
        # Get a batch of size dqn_batch_size from replay buffer to train the model
        dqn_batch = self.replay_buffer.next_batch()
        if dqn_batch is None:
          tf.logging.info('replay buffer not loaded enough yet...')
          time.sleep(60)
          continue
        # Run train step for Current DQN model and collect the results
        dqn_results = self.dqn.run_train_steps(self.dqn_sess, dqn_batch)
        # Run test step for Target DQN model and collect the results and monitor the difference in loss between the two
        dqn_target_results = self.dqn_target.run_test_steps(self.dqn_sess, x=dqn_batch._x, y=dqn_batch._y, return_loss=True)
        self.dqn_train_step = dqn_results['global_step']
        self.dqn_summary_writer.add_summary(dqn_results['summaries'], self.dqn_train_step) # write the summaries
        self.avg_dqn_loss.append(dqn_results['loss'])
        avg_dqn_target_loss.append(dqn_target_results['loss'])
        self.dqn_train_step = self.dqn_train_step + 1
        tf.logging.info('seconds for training dqn model: {}'.format(time.time()-_t))
        # UPDATING TARGET DDQN NETWORK WITH CURRENT MODEL
        with self.dqn_graph.as_default():
          current_model_weights = self.dqn_sess.run([self.dqn.model_trainables])[0] # get weights of current model
          self.dqn_target.run_update_weights(self.dqn_sess, self.dqn_train_step, current_model_weights) # update target model weights with current model weights
        tf.logging.info('DQN loss at step {}: {}'.format(self.dqn_train_step, np.mean(self.avg_dqn_loss)))
        tf.logging.info('DQN Target loss at step {}: {}'.format(self.dqn_train_step, np.mean(avg_dqn_target_loss)))
        # sleeping is required if you want the keyboard interuption to work
        time.sleep(FLAGS.dqn_sleep_time)
    except (KeyboardInterrupt, SystemExit):
      tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
      self.sv.stop()
      self.dqn_sv.stop()

  def watch_threads(self):
    """Watch example queue and batch queue threads and restart if dead."""
    while True:
      time.sleep(60)
      if not self.thrd_dqn_training.is_alive(): # if the thread is dead
        tf.logging.error('Found DQN Learning thread dead. Restarting.')
        self.thrd_dqn_training = Thread(target=self.dqn_training)
        self.thrd_dqn_training.daemon = True
        self.thrd_dqn_training.start()

  def run_eval(self):
    """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
    self.model.build_graph() # build the graph
    saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
    sess = tf.Session(config=util.get_config())

    if FLAGS.embedding:
      sess.run(tf.global_variables_initializer(),feed_dict={self.model.embedding_place:self.word_vector})
    eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
    self.summary_writer = tf.summary.FileWriter(eval_dir)

    if FLAGS.ac_training:
      tf.logging.info('DDQN building graph')
      t1 = time.time()
      dqn_graph = tf.Graph()
      with dqn_graph.as_default():
        self.dqn.build_graph() # build dqn graph
        tf.logging.info('building current network took {} seconds'.format(time.time()-t1))
        self.dqn_target.build_graph() # build dqn target graph
        tf.logging.info('building target network took {} seconds'.format(time.time()-t1))
        dqn_saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time
        dqn_sess = tf.Session(config=util.get_config())
      dqn_train_step = 0
      replay_buffer = ReplayBuffer(self.dqn_hps)

    running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = self.restore_best_eval_model()  # will hold the best loss achieved so far
    train_step = 0

    while True:
      _ = util.load_ckpt(saver, sess) # load a new checkpoint
      if FLAGS.ac_training:
        _ = util.load_dqn_ckpt(dqn_saver, dqn_sess) # load a new checkpoint
      processed_batch = 0
      avg_losses = []
      # evaluate for 100 * batch_size before comparing the loss
      # we do this due to memory constraint, best to run eval on different machines with large batch size
      while processed_batch < 100*FLAGS.batch_size:
        processed_batch += FLAGS.batch_size
        batch = self.batcher.next_batch() # get the next batch
        if FLAGS.ac_training:
          t0 = time.time()
          transitions = self.model.collect_dqn_transitions(sess, batch, train_step, batch.max_art_oovs) # len(batch_size * k * max_dec_steps)
          tf.logging.info('Q values collection time: {}'.format(time.time()-t0))
          with dqn_graph.as_default():
            # if using true Q-value to train DQN network,
            # we do this as the pre-training for the DQN network to get better estimates
            batch_len = len(transitions)
            b = ReplayBuffer.create_batch(self.dqn_hps, transitions,len(transitions), use_state_prime = True, max_art_oovs = batch.max_art_oovs)
            b_prime = ReplayBuffer.create_batch(self.dqn_hps, transitions,len(transitions), use_state_prime = True, max_art_oovs = batch.max_art_oovs)
            dqn_results = self.dqn.run_test_steps(sess=dqn_sess, x= b._x, return_best_action=True)
            q_estimates = dqn_results['estimates'] # shape (len(transitions), vocab_size)
            dqn_best_action = dqn_results['best_action']

            tf.logging.info('running test step on dqn_target')
            dqn_target_results = self.dqn_target.run_test_steps(dqn_sess, x= b_prime._x)
            q_vals_new_t = dqn_target_results['estimates'] # shape (len(transitions), vocab_size)

            # we need to expand the q_estimates to match the input batch max_art_oov
            q_estimates = np.concatenate([q_estimates,np.zeros((len(transitions),batch.max_art_oovs))],axis=-1)

            tf.logging.info('fixing the action q-estimates')
            for i, tr in enumerate(transitions):
              if tr.done:
                q_estimates[i][tr.action] = tr.reward
              else:
                q_estimates[i][tr.action] = tr.reward + FLAGS.gamma * q_vals_new_t[i][dqn_best_action[i]]
            if FLAGS.dqn_scheduled_sampling:
              tf.logging.info('scheduled sampling on q-estimates')
              q_estimates = self.scheduled_sampling(batch_len, FLAGS.sampling_probability, b._y_extended, q_estimates)
            if not FLAGS.calculate_true_q:
              # when we are not training DQN based on true Q-values
              # we need to update Q-values in our transitions based on this q_estimates we collected from DQN current network.
              for trans, q_val in zip(transitions,q_estimates):
                trans.q_values = q_val # each have the size vocab_extended
            q_estimates = np.reshape(q_estimates, [FLAGS.batch_size, FLAGS.k, FLAGS.max_dec_steps, -1]) # shape (batch_size, k, max_dec_steps, vocab_size_extended)
          tf.logging.info('run eval step on seq2seq model.')
          t0=time.time()
          results = self.model.run_eval_step(sess, batch, train_step, q_estimates)
          t1=time.time()
        else:
          tf.logging.info('run eval step on seq2seq model.')
          t0=time.time()
          results = self.model.run_eval_step(sess, batch, train_step)
          t1=time.time()

        tf.logging.info('experiment: {}'.format(FLAGS.exp_name))
        tf.logging.info('processed_batch: {}, seconds for batch: {}'.format(processed_batch, t1-t0))

        printer_helper = {}
        loss = printer_helper['pgen_loss']= results['pgen_loss']
        if FLAGS.coverage:
          printer_helper['coverage_loss'] = results['coverage_loss']
          if FLAGS.rl_training or FLAGS.ac_training:
            loss = printer_helper['rl_cov_total_loss']= results['reinforce_cov_total_loss']
          else:
            loss = printer_helper['pointer_cov_total_loss'] = results['pointer_cov_total_loss']
        if FLAGS.rl_training or FLAGS.ac_training:
          printer_helper['shared_loss'] = results['shared_loss']
          printer_helper['rl_loss'] = results['rl_loss']
          printer_helper['rl_avg_logprobs'] = results['rl_avg_logprobs']
        if FLAGS.rl_training:
          printer_helper['sampled_r'] = np.mean(results['sampled_sentence_r_values'])
          printer_helper['greedy_r'] = np.mean(results['greedy_sentence_r_values'])
          printer_helper['r_diff'] = printer_helper['sampled_r'] - printer_helper['greedy_r']
        if FLAGS.ac_training:
          printer_helper['dqn_loss'] = np.mean(self.avg_dqn_loss) if len(self.avg_dqn_loss) > 0 else 0

        for (k,v) in printer_helper.items():
          if not np.isfinite(v):
            raise Exception("{} is not finite. Stopping.".format(k))
          tf.logging.info('{}: {}\t'.format(k,v))

        # add summaries
        summaries = results['summaries']
        train_step = results['global_step']
        self.summary_writer.add_summary(summaries, train_step)

        # calculate running avg loss
        avg_losses.append(self.calc_running_avg_loss(np.asscalar(loss), running_avg_loss, train_step))
        tf.logging.info('-------------------------------------------')

      running_avg_loss = np.mean(avg_losses)
      tf.logging.info('==========================================')
      tf.logging.info('best_loss: {}\trunning_avg_loss: {}\t'.format(best_loss, running_avg_loss))
      tf.logging.info('==========================================')

      # If running_avg_loss is best so far, save this checkpoint (early stopping).
      # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
      if best_loss is None or running_avg_loss < best_loss:
        tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
        saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
        best_loss = running_avg_loss

      # flush the summary writer every so often
      if train_step % 100 == 0:
        self.summary_writer.flush()
      #time.sleep(600) # run eval every 10 minute

  def main(self, unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
      raise Exception("Problem with flags: %s" % unused_argv)

    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
    tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    flags = getattr(FLAGS,"__flags")

    if not os.path.exists(FLAGS.log_root):
      if FLAGS.mode=="train":
        os.makedirs(FLAGS.log_root)
      else:
        raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    fw = open('{}/config.txt'.format(FLAGS.log_root), 'w')
    for k, v in flags.items():
      fw.write('{}\t{}\n'.format(k, v))
    fw.close()

    self.vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size) # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if FLAGS.mode == 'decode':
      FLAGS.batch_size = FLAGS.beam_size

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and FLAGS.mode!='decode':
      raise Exception("The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs

    hparam_list = ['mode', 'lr', 'gpu_num',
    #'sampled_greedy_flag', 
    'gamma', 'eta', 
    'fixed_eta', 'reward_function', 'intradecoder', 
    'use_temporal_attention', 'ac_training','rl_training', 'matrix_attention', 'calculate_true_q',
    'enc_hidden_dim', 'dec_hidden_dim', 'k', 
    'scheduled_sampling', 'sampling_probability','fixed_sampling_probability',
    'alpha', 'hard_argmax', 'greedy_scheduled_sampling',
    'adagrad_init_acc', 'rand_unif_init_mag', 
    'trunc_norm_init_std', 'max_grad_norm', 
    'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps',
    'dqn_scheduled_sampling', 'dqn_sleep_time', 'E2EBackProp',
    'coverage', 'cov_loss_wt', 'pointer_gen']
    hps_dict = {}
    for key,val in flags.items(): # for each flag
      if key in hparam_list: # if it's in the list
        hps_dict[key] = val.value # add it to the dict
    if FLAGS.ac_training:
      hps_dict.update({'dqn_input_feature_len':(FLAGS.dec_hidden_dim)})
    self.hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    # creating all the required parameters for DDQN model.
    if FLAGS.ac_training:
      hparam_list = ['lr', 'dqn_gpu_num', 
      'dqn_layers', 
      'dqn_replay_buffer_size', 
      'dqn_batch_size', 
      'dqn_target_update',
      'dueling_net',
      'dqn_polyak_averaging',
      'dqn_sleep_time',
      'dqn_scheduled_sampling',
      'max_grad_norm']
      hps_dict = {}
      for key,val in flags.items(): # for each flag
        if key in hparam_list: # if it's in the list
          hps_dict[key] = val.value # add it to the dict
      hps_dict.update({'dqn_input_feature_len':(FLAGS.dec_hidden_dim)})
      hps_dict.update({'vocab_size':self.vocab.size()})
      self.dqn_hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    # Create a batcher object that will create minibatches of data
    self.batcher = Batcher(FLAGS.data_path, self.vocab, self.hps, single_pass=FLAGS.single_pass, decode_after=FLAGS.decode_after)

    tf.set_random_seed(111) # a seed value for randomness

    if self.hps.mode == 'train':
      print("creating model...")
      self.model = SummarizationModel(self.hps, self.vocab)
      if FLAGS.ac_training:
        # current DQN with paramters \Psi
        self.dqn = DQN(self.dqn_hps,'current')
        # target DQN with paramters \Psi^{\prime}
        self.dqn_target = DQN(self.dqn_hps,'target')
      self.setup_training()
    elif self.hps.mode == 'eval':
      self.model = SummarizationModel(self.hps, self.vocab)
      if FLAGS.ac_training:
        self.dqn = DQN(self.dqn_hps,'current')
        self.dqn_target = DQN(self.dqn_hps,'target')
      self.run_eval()
    elif self.hps.mode == 'decode':
      decode_model_hps = self.hps  # This will be the hyperparameters for the decoder model
      decode_model_hps = self.hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
      model = SummarizationModel(decode_model_hps, self.vocab)
      if FLAGS.ac_training:
        # We need our target DDQN network for collecting Q-estimation at each decoder step.
        dqn_target = DQN(self.dqn_hps,'target')
      else:
        dqn_target = None
      decoder = BeamSearchDecoder(model, self.batcher, self.vocab, dqn = dqn_target)
      decoder.decode() # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
    else:
      raise ValueError("The 'mode' flag must be one of train/eval/decode")

  # Scheduled sampling used for either selecting true Q-estimates or the DDQN estimation
  # based on https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/ScheduledEmbeddingTrainingHelper
  def scheduled_sampling(self, batch_size, sampling_probability, true, estimate):
    with variable_scope.variable_scope("ScheduledEmbedding"):
      # Return -1s where we do not sample, and sample_ids elsewhere
      select_sampler = bernoulli.Bernoulli(probs=sampling_probability, dtype=tf.bool)
      select_sample = select_sampler.sample(sample_shape=batch_size)
      sample_ids = array_ops.where(
                  select_sample,
                  tf.range(batch_size),
                  gen_array_ops.fill([batch_size], -1))
      where_sampling = math_ops.cast(
          array_ops.where(sample_ids > -1), tf.int32)
      where_not_sampling = math_ops.cast(
          array_ops.where(sample_ids <= -1), tf.int32)
      _estimate = array_ops.gather_nd(estimate, where_sampling)
      _true = array_ops.gather_nd(true, where_not_sampling)

      base_shape = array_ops.shape(true)
      result1 = array_ops.scatter_nd(indices=where_sampling, updates=_estimate, shape=base_shape)
      result2 = array_ops.scatter_nd(indices=where_not_sampling, updates=_true, shape=base_shape)
      result = result1 + result2
      return result1 + result2

def main(unused_argv):
  seq2seq = Seq2Seq()
  seq2seq.main(unused_argv)

if __name__ == '__main__':
  tf.app.run()
