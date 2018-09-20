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

"""This file defines the decoder"""

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.ops.distributions import bernoulli
from rouge_tensor import rouge_l_fscore

FLAGS = tf.app.flags.FLAGS

def print_shape(str, var):
  tf.logging.info('shape of {}: {}'.format(str, [k for k in var.get_shape()]))


def _calc_final_dist(_hps, v_size, _max_art_oovs, _enc_batch_extend_vocab, p_gen, vocab_dist, attn_dist):
  """Calculate the final distribution, for the pointer-generator model
  Args:
    vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
    attn_dists: The attention distributions. List length max_dec_steps of (batch_size, max_enc_steps) arrays

  Returns:
    final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
  """
  with tf.variable_scope('final_distribution'):
    # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
    vocab_dist = p_gen * vocab_dist
    attn_dist = (1-p_gen) * attn_dist

    # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
    extended_vsize = v_size + _max_art_oovs # the maximum (over the batch) size of the extended vocabulary
    extra_zeros = tf.zeros((_hps.batch_size, _max_art_oovs))
    vocab_dists_extended = tf.concat(axis=1, values=[vocab_dist, extra_zeros]) # list length max_dec_steps of shape (batch_size, extended_vsize)

    # Project the values in the attention distributions onto the appropriate entries in the final distributions
    # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
    # This is done for each decoder timestep.
    # This is fiddly; we use tf.scatter_nd to do the projection
    batch_nums = tf.range(0, limit=_hps.batch_size) # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
    attn_len = tf.shape(_enc_batch_extend_vocab)[1] # number of states we attend over
    batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
    indices = tf.stack( (batch_nums, _enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
    shape = [_hps.batch_size, extended_vsize]
    attn_dists_projected = tf.scatter_nd(indices, attn_dist, shape) # list length max_dec_steps (batch_size, extended_vsize)

    # Add the vocab distributions and the copy distributions together to get the final distributions
    # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
    # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
    final_dist = vocab_dists_extended + attn_dists_projected
    final_dist +=1e-15 # for cases where we have zero in the final dist, especially for oov words
    dist_sums = tf.reduce_sum(final_dist, axis=1)
    final_dist = final_dist / tf.reshape(dist_sums, [-1, 1]) # re-normalize

  return final_dist

# Note: this function is based on tf.contrib.legacy_seq2seq_attention_decoder, which is now outdated.
# In the future, it would make more sense to write variants on the attention mechanism using the new seq2seq library for tensorflow 1.0: https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention
def attention_decoder(_hps, 
  v_size, 
  _max_art_oovs, 
  _enc_batch_extend_vocab, 
  emb_dec_inputs,
  target_batch,
  _dec_in_state, 
  _enc_states, 
  enc_padding_mask, 
  dec_padding_mask, 
  cell, 
  embedding, 
  sampling_probability,
  alpha,
  unk_id,
  initial_state_attention=False,
  pointer_gen=True, 
  use_coverage=False, 
  prev_coverage=None, 
  prev_decoder_outputs=[], 
  prev_encoder_es = []):
  """
  Args:
    _hps: parameter of the models.
    v_size: vocab size.
    _max_art_oovs: size of the oov tokens in current batch.
    _enc_batch_extend_vocab: encoder extended vocab batch.
    emb_dec_inputs: A list of 2D Tensors [batch_size x emb_dim].
    target_batch: The indices of the target words. shape (max_dec_steps, batch_size)
    _dec_in_state: 2D Tensor [batch_size x cell.state_size].
    _enc_states: 3D Tensor [batch_size x max_enc_steps x attn_size].
    enc_padding_mask: 2D Tensor [batch_size x max_enc_steps] containing 1s and 0s; indicates which of the encoder locations are padding (0) or a real token (1).
    dec_padding_mask: 2D Tensor [batch_size x max_dec_steps] containing 1s and 0s; indicates which of the decoder locations are padding (0) or a real token (1).
    cell: rnn_cell.RNNCell defining the cell function and size.
    embedding: embedding matrix [vocab_size, emb_dim].
    sampling_probability: sampling probability for scheduled sampling.
    alpha: soft-argmax argument.
    initial_state_attention:
      Note that this attention decoder passes each decoder input through a linear layer with the previous step's context vector to get a modified version of the input. If initial_state_attention is False, on the first decoder step the "previous context vector" is just a zero vector. If initial_state_attention is True, we use _dec_in_state to (re)calculate the previous step's context vector. We set this to False for train/eval mode (because we call attention_decoder once for all decoder steps) and True for decode mode (because we call attention_decoder once for each decoder step).
    pointer_gen: boolean. If True, calculate the generation probability p_gen for each decoder step.
    use_coverage: boolean. If True, use coverage mechanism.
    prev_coverage:
      If not None, a tensor with shape (batch_size, max_enc_steps). The previous step's coverage vector. This is only not None in decode mode when using coverage.
    prev_decoder_outputs: if not empty, a tensor of (len(prev_decoder_steps), batch_size, hidden_dim). The previous decoder output used for calculating the intradecoder attention during decode mode
    prev_encoder_es: if not empty, a tensor of (len(prev_encoder_es), batch_size, hidden_dim). The previous attention vector used for calculating the temporal attention during decode mode.
  Returns:
    outputs: A list of the same length as emb_dec_inputs of 2D Tensors of
      shape [batch_size x cell.output_size]. The output vectors.
    state: The final state of the decoder. A tensor shape [batch_size x cell.state_size].
    attn_dists: A list containing tensors of shape (batch_size,max_enc_steps).
      The attention distributions for each decoder step.
    p_gens: List of length emb_dim, containing tensors of shape [batch_size, 1]. The values of p_gen for each decoder step. Empty list if pointer_gen=False.
    coverage: Coverage vector on the last step computed. None if use_coverage=False.
    vocab_scores: vocab distribution.
    final_dists: final output distribution.
    samples: contains sampled tokens.
    greedy_search_samples: contains greedy tokens.
    temporal_e: contains temporal attention.
  """
  with variable_scope.variable_scope("attention_decoder") as scope:
    batch_size = _enc_states.get_shape()[0] # if this line fails, it's because the batch size isn't defined
    attn_size = _enc_states.get_shape()[2] # if this line fails, it's because the attention length isn't defined
    emb_size = emb_dec_inputs[0].get_shape()[1] # if this line fails, it's because the embedding isn't defined
    decoder_attn_size = _dec_in_state.c.get_shape()[1]
    tf.logging.info("batch_size %i, attn_size: %i, emb_size: %i", batch_size, attn_size, emb_size)
    # Reshape _enc_states (need to insert a dim)
    _enc_states = tf.expand_dims(_enc_states, axis=2) # now is shape (batch_size, max_enc_steps, 1, attn_size)

    # To calculate attention, we calculate
    #   v^T tanh(W_h h_i + W_s s_t + b_attn)
    # where h_i is an encoder state, and s_t a decoder state.
    # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
    # We set it to be equal to the size of the encoder states.
    attention_vec_size = attn_size

    # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
    if _hps.matrix_attention:
      w_attn = variable_scope.get_variable("w_attn", [attention_vec_size, attention_vec_size])
      if _hps.intradecoder:
        w_dec_attn = variable_scope.get_variable("w_dec_attn", [decoder_attn_size, decoder_attn_size])
    else:
      W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
      v = variable_scope.get_variable("v", [attention_vec_size])
      encoder_features = nn_ops.conv2d(_enc_states, W_h, [1, 1, 1, 1], "SAME") # shape (batch_size,max_enc_steps,1,attention_vec_size)
    if _hps.intradecoder:
      W_h_d = variable_scope.get_variable("W_h_d", [1, 1, decoder_attn_size, decoder_attn_size])
      v_d = variable_scope.get_variable("v_d", [decoder_attn_size])

    # Get the weight vectors v and w_c (w_c is for coverage)
    if use_coverage:
      with variable_scope.variable_scope("coverage"):
        w_c = variable_scope.get_variable("w_c", [1, 1, 1, attention_vec_size])

    if prev_coverage is not None: # for beam search mode with coverage
      # reshape from (batch_size, max_enc_steps) to (batch_size, max_enc_steps, 1, 1)
      prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage,2),3)

    def attention(decoder_state, temporal_e, coverage=None):
      """Calculate the context vector and attention distribution from the decoder state.

      Args:
        decoder_state: state of the decoder
        temporal_e: store previous attentions for temporal attention mechanism
        coverage: Optional. Previous timestep's coverage vector, shape (batch_size, max_enc_steps, 1, 1).

      Returns:
        context_vector: weighted sum of _enc_states
        attn_dist: attention distribution
        coverage: new coverage vector. shape (batch_size, max_enc_steps, 1, 1)
        masked_e: store the attention score for temporal attention mechanism.
      """
      with variable_scope.variable_scope("Attention"):
        # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
        decoder_features = linear(decoder_state, attention_vec_size, True) # shape (batch_size, attention_vec_size)
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_vec_size)

        # We can't have coverage with matrix attention
        if not _hps.matrix_attention and use_coverage and coverage is not None: # non-first step of coverage
          # Multiply coverage vector by w_c to get coverage_features.
          coverage_features = nn_ops.conv2d(coverage, w_c, [1, 1, 1, 1], "SAME") # c has shape (batch_size, max_enc_steps, 1, attention_vec_size)
          # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
          e_not_masked = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features + coverage_features), [2, 3])  # shape (batch_size,max_enc_steps)
          masked_e = nn_ops.softmax(e_not_masked) * enc_padding_mask # (batch_size, max_enc_steps)
          masked_sums = tf.reduce_sum(masked_e, axis=1) # shape (batch_size)
          masked_e = masked_e / tf.reshape(masked_sums, [-1, 1])
          # Equation 3 in 
          if _hps.use_temporal_attention:
            try:
              len_temporal_e = temporal_e.get_shape()[0]
            except:
              len_temporal_e = 0
            if len_temporal_e==0:
              attn_dist = masked_e
            else:
              masked_sums = tf.reduce_sum(temporal_e,axis=0)+1e-10 # if it's zero due to masking we set it to a small value
              attn_dist = masked_e / masked_sums # (batch_size, max_enc_steps)
          else:
            attn_dist = masked_e
          masked_attn_sums = tf.reduce_sum(attn_dist, axis=1)
          attn_dist = attn_dist / tf.reshape(masked_attn_sums, [-1, 1]) # re-normalize
          # Update coverage vector
          coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
        else:
          if _hps.matrix_attention:
            # Calculate h_d * W_attn * h_i, equation 2 in https://arxiv.org/pdf/1705.04304.pdf
            _dec_attn = tf.unstack(tf.matmul(tf.squeeze(decoder_features,axis=[1,2]),w_attn),axis=0) # batch_size * (attention_vec_size)
            _enc_states_lst = tf.unstack(tf.squeeze(_enc_states,axis=2),axis=0) # batch_size * (max_enc_steps, attention_vec_size)

            e_not_masked = tf.squeeze(tf.stack([tf.matmul(tf.reshape(_dec,[1,-1]), tf.transpose(_enc)) for _dec, _enc in zip(_dec_attn,_enc_states_lst)]),axis=1) # (batch_size, max_enc_steps)
            masked_e = tf.exp(e_not_masked * enc_padding_mask) # (batch_size, max_enc_steps)
          else:
            # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
            e_not_masked = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features), [2, 3]) # calculate e, (batch_size, max_enc_steps)
            masked_e = nn_ops.softmax(e_not_masked) * enc_padding_mask # (batch_size, max_enc_steps)
            masked_sums = tf.reduce_sum(masked_e, axis=1) # shape (batch_size)
            masked_e = masked_e / tf.reshape(masked_sums, [-1, 1])
          if _hps.use_temporal_attention:
            try:
              len_temporal_e = temporal_e.get_shape()[0]
            except:
              len_temporal_e = 0
            if len_temporal_e==0:
              attn_dist = masked_e
            else:
              masked_sums = tf.reduce_sum(temporal_e,axis=0)+1e-10 # if it's zero due to masking we set it to a small value
              attn_dist = masked_e / masked_sums # (batch_size, max_enc_steps)
          else:
            attn_dist = masked_e
          # Calculate attention distribution
          masked_attn_sums = tf.reduce_sum(attn_dist, axis=1)
          attn_dist = attn_dist / tf.reshape(masked_attn_sums, [-1, 1]) # re-normalize

          if use_coverage: # first step of training
            coverage = tf.expand_dims(tf.expand_dims(attn_dist,2),2) # initialize coverage

        # Calculate the context vector from attn_dist and _enc_states
        context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * _enc_states, [1, 2]) # shape (batch_size, attn_size).
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])

      return context_vector, attn_dist, coverage, masked_e

    def intra_decoder_attention(decoder_state, outputs):
      """Calculate the context vector and attention distribution from the decoder state.

      Args:
        decoder_state: state of the decoder
        outputs: list of decoder states for implementing intra-decoder mechanism, len(decoder_states) * (batch_size, hidden_dim)
      Returns:
        context_decoder_vector: weighted sum of _dec_states
        decoder_attn_dist: intra-decoder attention distribution
      """
      attention_dec_vec_size = attn_dec_size = decoder_state.c.get_shape()[1] # hidden_dim
      try:
        len_dec_states = outputs.get_shape()[0]
      except:
        len_dec_states = 0
      attention_dec_vec_size = attn_dec_size = decoder_state.c.get_shape()[1] # hidden_dim
      _decoder_states = tf.expand_dims(tf.reshape(outputs,[batch_size,-1,attn_dec_size]), axis=2) # now is shape (batch_size,len(decoder_states), 1, attn_size)
      _prev_decoder_features = nn_ops.conv2d(_decoder_states, W_h_d, [1, 1, 1, 1], "SAME") # shape (batch_size,len(decoder_states),1,attention_vec_size)
      with variable_scope.variable_scope("DecoderAttention"):
        # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
        try:
          decoder_features = linear(decoder_state, attention_dec_vec_size, True) # shape (batch_size, attention_vec_size)
          decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1) # reshape to (batch_size, 1, 1, attention_dec_vec_size)
          # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
          if _hps.matrix_attention:
            # Calculate h_d * W_attn * h_d, equation 6 in https://arxiv.org/pdf/1705.04304.pdf
            _dec_attn = tf.matmul(tf.squeeze(decoder_features),w_dec_attn) # (batch_size, decoder_attn_size)
            _dec_states_lst = tf.unstack(tf.reshape(_prev_decoder_features,[batch_size,-1,decoder_attn_size])) # batch_size * (len(decoder_states), decoder_attn_size)
            e_not_masked = tf.reshape(tf.stack([tf.matmul(_dec_attn, tf.transpose(k)) for k in _dec_states_lst]),[batch_size,-1]) # (batch_size, len(decoder_states))
            masked_e = tf.exp(e_not_masked * dec_padding_mask[:,:len_dec_states]) # (batch_size, len(decoder_states))
          else:
            # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
            e_not_masked = math_ops.reduce_sum(v_d * math_ops.tanh(_prev_decoder_features + decoder_features), [2, 3]) # calculate e, (batch_size,len(decoder_states))
            masked_e = nn_ops.softmax(e_not_masked) * dec_padding_mask[:,:len_dec_states] # (batch_size,len(decoder_states))
          if len_dec_states <= 1:
            masked_e = array_ops.ones([batch_size,1]) # first step is filled with equal values
          masked_sums = tf.reshape(tf.reduce_sum(masked_e,axis=1),[-1,1]) # (batch_size,1), # if it's zero due to masking we set it to a small value
          decoder_attn_dist = masked_e / masked_sums # (batch_size,len(decoder_states))
          context_decoder_vector = math_ops.reduce_sum(array_ops.reshape(decoder_attn_dist, [batch_size, -1, 1, 1]) * _decoder_states, [1, 2]) # (batch_size, attn_size)
          context_decoder_vector = array_ops.reshape(context_decoder_vector, [-1, attn_dec_size]) # (batch_size, attn_size)
        except:
          return array_ops.zeros([batch_size, decoder_attn_size]), array_ops.zeros([batch_size, 0])
      return context_decoder_vector, decoder_attn_dist

    outputs = []
    temporal_e = []
    attn_dists = []
    vocab_scores = []
    vocab_dists = []
    final_dists = []
    p_gens = []
    samples = [] # this holds the words chosen by sampling based on the final distribution for each decoding step, list of max_dec_steps of (batch_size, 1)
    greedy_search_samples = [] # this holds the words chosen by greedy search (taking the max) on the final distribution for each decoding step, list of max_dec_steps of (batch_size, 1)
    sampling_rewards = [] # list of size max_dec_steps (batch_size, k)
    greedy_rewards = [] # list of size max_dec_steps (batch_size, k)
    state = _dec_in_state
    coverage = prev_coverage # initialize coverage to None or whatever was passed in
    context_vector = array_ops.zeros([batch_size, attn_size])
    context_decoder_vector = array_ops.zeros([batch_size, decoder_attn_size])
    context_vector.set_shape([None, attn_size])  # Ensure the second shape of attention vectors is set.
    if initial_state_attention: # true in decode mode
      # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
      context_vector, _, coverage, _ = attention(_dec_in_state, tf.stack(prev_encoder_es,axis=0), coverage) # in decode mode, this is what updates the coverage vector
      if _hps.intradecoder:
        context_decoder_vector, _ = intra_decoder_attention(_dec_in_state, tf.stack(prev_decoder_outputs,axis=0))
    for i, inp in enumerate(emb_dec_inputs):
      tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(emb_dec_inputs))
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      if _hps.mode in ['train','eval'] and _hps.scheduled_sampling and i > 0: # start scheduled sampling after we received the first decoder's output
        # modify the input to next decoder using scheduled sampling
        if FLAGS.scheduled_sampling_final_dist:
          inp = scheduled_sampling(_hps, sampling_probability, final_dist, embedding, inp, alpha)
        else:
          inp = scheduled_sampling_vocab_dist(_hps, sampling_probability, vocab_dist, embedding, inp, alpha)

      # Merge input and previous attentions into one vector x of the same size as inp
      emb_dim = inp.get_shape().with_rank(2)[1]
      if emb_dim is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)

      x = linear([inp] + [context_vector], emb_dim, True)
      # Run the decoder RNN cell. cell_output = decoder state
      cell_output, state = cell(x, state)

      # Run the attention mechanism.
      if i == 0 and initial_state_attention:  # always true in decode mode
        with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True): # you need this because you've already run the initial attention(...) call
          context_vector, attn_dist, _, masked_e = attention(state, tf.stack(prev_encoder_es,axis=0), coverage) # don't allow coverage to update
          if _hps.intradecoder:
            context_decoder_vector, _ = intra_decoder_attention(state, tf.stack(prev_decoder_outputs,axis=0))
      else:
        context_vector, attn_dist, coverage, masked_e = attention(state, tf.stack(temporal_e,axis=0), coverage)
        if _hps.intradecoder:
          context_decoder_vector, _ = intra_decoder_attention(state, tf.stack(outputs,axis=0))
      attn_dists.append(attn_dist)
      temporal_e.append(masked_e)

      with variable_scope.variable_scope("combined_context"):
        if _hps.intradecoder:
          context_vector = linear([context_vector] + [context_decoder_vector], attn_size, False)
      # Calculate p_gen
      if pointer_gen:
        with tf.variable_scope('calculate_pgen'):
          p_gen = linear([context_vector, state.c, state.h, x], 1, True) # Tensor shape (batch_size, 1)
          p_gen = tf.sigmoid(p_gen)
          p_gens.append(p_gen)

      # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
      # This is V[s_t, h*_t] + b in the paper
      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + [context_vector], cell.output_size, True)
      outputs.append(output)

      # Add the output projection to obtain the vocabulary distribution
      with tf.variable_scope('output_projection'):
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        trunc_norm_init = tf.truncated_normal_initializer(stddev=_hps.trunc_norm_init_std)
        w_out = tf.get_variable('w', [_hps.dec_hidden_dim, v_size], dtype=tf.float32, initializer=trunc_norm_init)
        #w_t_out = tf.transpose(w)
        v_out = tf.get_variable('v', [v_size], dtype=tf.float32, initializer=trunc_norm_init)
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        if FLAGS.share_decoder_weights: # Eq. 13 in https://arxiv.org/pdf/1705.04304.pdf
          w_out = tf.transpose(
            math_ops.tanh(linear([embedding] + [tf.transpose(w_out)], _hps.dec_hidden_dim, bias=False)))
        score = tf.nn.xw_plus_b(output, w_out, v_out)
        if _hps.scheduled_sampling and not _hps.greedy_scheduled_sampling:
          # Gumbel reparametrization trick: https://arxiv.org/abs/1704.06970
          U = tf.random_uniform(score.get_shape(),10e-12,(1-10e-12)) # add a small number to avoid log(0)
          G = -tf.log(-tf.log(U))
          score = score + G
        vocab_scores.append(score) # apply the linear layer
        vocab_dist = tf.nn.softmax(score)
        vocab_dists.append(vocab_dist) # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.

      # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
      if _hps.pointer_gen:
        final_dist = _calc_final_dist(_hps, v_size, _max_art_oovs, _enc_batch_extend_vocab, p_gen, vocab_dist,
                                      attn_dist)
      else: # final distribution is just vocabulary distribution
        final_dist = vocab_dist
      final_dists.append(final_dist)

      # get the sampled token and greedy token
      # this will take the final_dist and sample from it for a total count of k (k samples)
      one_hot_k_samples = tf.distributions.Multinomial(total_count=1., probs=final_dist).sample(
        _hps.k)  # sample k times according to https://arxiv.org/pdf/1705.04304.pdf, size (k, batch_size, extended_vsize)
      k_argmax = tf.argmax(one_hot_k_samples, axis=2, output_type=tf.int32) # (k, batch_size)
      k_sample = tf.transpose(k_argmax) # shape (batch_size, k)
      greedy_search_prob, greedy_search_sample = tf.nn.top_k(final_dist, k=_hps.k) # (batch_size, k)
      greedy_search_samples.append(greedy_search_sample)
      samples.append(k_sample)
      if FLAGS.use_discounted_rewards:
        _sampling_rewards = []
        _greedy_rewards = []
        for _ in range(_hps.k):
          rl_fscore = tf.reshape(rouge_l_fscore(tf.transpose(tf.stack(samples)[:, :, _]), target_batch),
                                 [-1, 1])  # shape (batch_size, 1)
          _sampling_rewards.append(tf.reshape(rl_fscore, [-1, 1]))
          rl_fscore = tf.reshape(rouge_l_fscore(tf.transpose(tf.stack(greedy_search_samples)[:, :, _]), target_batch),
                                 [-1, 1])  # shape (batch_size, 1)
          _greedy_rewards.append(tf.reshape(rl_fscore, [-1, 1]))
        sampling_rewards.append(tf.squeeze(tf.stack(_sampling_rewards, axis=1), axis = -1)) # (batch_size, k)
        greedy_rewards.append(tf.squeeze(tf.stack(_greedy_rewards, axis=1), axis = -1))  # (batch_size, k)

    if FLAGS.use_discounted_rewards:
      sampling_rewards = tf.stack(sampling_rewards)
      greedy_rewards = tf.stack(greedy_rewards)
    else:
      _sampling_rewards = []
      _greedy_rewards = []
      for _ in range(_hps.k):
        rl_fscore = rouge_l_fscore(tf.transpose(tf.stack(samples)[:, :, _]), target_batch) # shape (batch_size, 1)
        _sampling_rewards.append(tf.reshape(rl_fscore, [-1, 1]))
        rl_fscore = rouge_l_fscore(tf.transpose(tf.stack(greedy_search_samples)[:, :, _]), target_batch)  # shape (batch_size, 1)
        _greedy_rewards.append(tf.reshape(rl_fscore, [-1, 1]))
      sampling_rewards = tf.squeeze(tf.stack(_sampling_rewards, axis=1), axis=-1) # (batch_size, k)
      greedy_rewards = tf.squeeze(tf.stack(_greedy_rewards, axis=1), axis=-1) # (batch_size, k)
    # If using coverage, reshape it
    if coverage is not None:
      coverage = array_ops.reshape(coverage, [batch_size, -1])

  return (
  outputs, state, attn_dists, p_gens, coverage, vocab_scores, final_dists, samples, greedy_search_samples, temporal_e,
  sampling_rewards, greedy_rewards)

def scheduled_sampling(hps, sampling_probability, output, embedding, inp, alpha = 0):
  # borrowed ideas from https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/ScheduledEmbeddingTrainingHelper
  vocab_size = embedding.get_shape()[0]

  def soft_argmax(alpha, _output):
    new_oov_scores = tf.reshape(_output[:, 0] + tf.reduce_sum(_output[:, vocab_size:], axis=1),
                                [-1, 1])  # add score for all OOV to the UNK score
    _output = tf.concat([new_oov_scores, _output[:, 1:vocab_size]], axis=1) # select only the vocab_size outputs
    _output = _output / tf.reshape(tf.reduce_sum(output, axis=1), [-1, 1]) # re-normalize scores

    #alpha_exp = tf.exp(alpha * _output) # (batch_size, vocab_size)
    #one_hot_scores = alpha_exp / tf.reshape(tf.reduce_sum(alpha_exp, axis=1),[-1,1]) #(batch_size, vocab_size)
    one_hot_scores = tf.nn.softmax((alpha * _output))
    return one_hot_scores

  def soft_top_k(alpha, _output, K):
    copy = tf.identity(_output)
    p = []
    arg_top_k = []
    for k in range(K):
      sargmax = soft_argmax(alpha, copy)
      copy = (1-sargmax)* copy
      p.append(tf.reduce_sum(sargmax * _output, axis=1))
      arg_top_k.append(sargmax)

    return tf.stack(p, axis=1), tf.stack(arg_top_k)

  with variable_scope.variable_scope("ScheduledEmbedding"):
    # Return -1s where we did not sample, and sample_ids elsewhere
    select_sampler = bernoulli.Bernoulli(probs=sampling_probability, dtype=tf.bool)
    select_sample = select_sampler.sample(sample_shape=hps.batch_size)
    sample_id_sampler = categorical.Categorical(probs=output) # equals to argmax{ Multinomial(output, total_count=1) }, our greedy search selection
    sample_ids = array_ops.where(
            select_sample,
            sample_id_sampler.sample(seed=123),
            gen_array_ops.fill([hps.batch_size], -1))

    where_sampling = math_ops.cast(
        array_ops.where(sample_ids > -1), tf.int32)
    where_not_sampling = math_ops.cast(
        array_ops.where(sample_ids <= -1), tf.int32)

    if hps.greedy_scheduled_sampling:
      sample_ids = tf.argmax(output, axis=1, output_type=tf.int32)

    sample_ids_sampling = array_ops.gather_nd(sample_ids, where_sampling)

    cond = tf.less(sample_ids_sampling, vocab_size) # replace oov with unk
    sample_ids_sampling = tf.cast(cond, tf.int32) * sample_ids_sampling
    inputs_not_sampling = array_ops.gather_nd(inp, where_not_sampling)

    if hps.E2EBackProp:
      if hps.hard_argmax:
        greedy_search_prob, greedy_search_sample = tf.nn.top_k(output, k=hps.k) # (batch_size, k)
        greedy_search_prob_normalized = greedy_search_prob/tf.reshape(tf.reduce_sum(greedy_search_prob,axis=1),[-1,1])

        cond = tf.less(greedy_search_sample, vocab_size) # replace oov with unk
        greedy_search_sample = tf.cast(cond, tf.int32) * greedy_search_sample

        greedy_embedding = tf.nn.embedding_lookup(embedding, greedy_search_sample)
        normalized_embedding = tf.multiply(tf.reshape(greedy_search_prob_normalized,[hps.batch_size,hps.k,1]), greedy_embedding)
        e2e_embedding = tf.reduce_mean(normalized_embedding,axis=1)
      else:
        e = []
        greedy_search_prob, greedy_search_sample = soft_top_k(alpha, output,
                                                              K=hps.k)  # (batch_size, k), (k, batch_size, vocab_size)
        greedy_search_prob_normalized = greedy_search_prob / tf.reshape(tf.reduce_sum(greedy_search_prob, axis=1),
                                                                        [-1, 1])

        for _ in range(hps.k):
          a_k = greedy_search_sample[_]
          e_k = tf.matmul(tf.reshape(greedy_search_prob_normalized[:,_],[-1,1]) * a_k, embedding)
          e.append(e_k)
        e2e_embedding = tf.reduce_sum(e, axis=0) # (batch_size, emb_dim)
      sampled_next_inputs = array_ops.gather_nd(e2e_embedding, where_sampling)
    else:
      if hps.hard_argmax:
        sampled_next_inputs = tf.nn.embedding_lookup(embedding, sample_ids_sampling)
      else: # using soft armax (greedy) proposed in: https://arxiv.org/abs/1704.06970
        #alpha_exp = tf.exp(alpha * (output_not_extended + G)) # (batch_size, vocab_size)
        #one_hot_scores = alpha_exp / tf.reduce_sum(alpha_exp, axis=1) #(batch_size, vocab_size)
        one_hot_scores = soft_argmax(alpha, output) #(batch_size, vocab_size)
        soft_argmax_embedding = tf.matmul(one_hot_scores, embedding) #(batch_size, emb_size)
        sampled_next_inputs = array_ops.gather_nd(soft_argmax_embedding, where_sampling)

    base_shape = array_ops.shape(inp)
    result1 = array_ops.scatter_nd(indices=where_sampling, updates=sampled_next_inputs, shape=base_shape)
    result2 = array_ops.scatter_nd(indices=where_not_sampling, updates=inputs_not_sampling, shape=base_shape)
    return result1 + result2

def scheduled_sampling_vocab_dist(hps, sampling_probability, output, embedding, inp, alpha = 0):
  # borrowed ideas from https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/ScheduledEmbeddingTrainingHelper

  def soft_argmax(alpha, output):
    #alpha_exp = tf.exp(alpha * output) # (batch_size, vocab_size)
    #one_hot_scores = alpha_exp / tf.reshape(tf.reduce_sum(alpha_exp, axis=1),[-1,1]) #(batch_size, vocab_size)
    one_hot_scores = tf.nn.softmax(alpha * output)
    return one_hot_scores

  def soft_top_k(alpha, output, K):
    copy = tf.identity(output)
    p = []
    arg_top_k = []
    for k in range(K):
      sargmax = soft_argmax(alpha, copy)
      copy = (1-sargmax)* copy
      p.append(tf.reduce_sum(sargmax * output, axis=1))
      arg_top_k.append(sargmax)

    return tf.stack(p, axis=1), tf.stack(arg_top_k)

  with variable_scope.variable_scope("ScheduledEmbedding"):
    # Return -1s where we did not sample, and sample_ids elsewhere
    select_sampler = bernoulli.Bernoulli(probs=sampling_probability, dtype=tf.bool)
    select_sample = select_sampler.sample(sample_shape=hps.batch_size)
    sample_id_sampler = categorical.Categorical(probs=output) # equals to argmax{ Multinomial(output, total_count=1) }, our greedy search selection
    sample_ids = array_ops.where(
            select_sample,
            sample_id_sampler.sample(seed=123),
            gen_array_ops.fill([hps.batch_size], -1))

    where_sampling = math_ops.cast(
        array_ops.where(sample_ids > -1), tf.int32)
    where_not_sampling = math_ops.cast(
        array_ops.where(sample_ids <= -1), tf.int32)

    if hps.greedy_scheduled_sampling:
      sample_ids = tf.argmax(output, axis=1, output_type=tf.int32)

    sample_ids_sampling = array_ops.gather_nd(sample_ids, where_sampling)
    inputs_not_sampling = array_ops.gather_nd(inp, where_not_sampling)

    if hps.E2EBackProp:
      if hps.hard_argmax:
        greedy_search_prob, greedy_search_sample = tf.nn.top_k(output, k=hps.k) # (batch_size, k)
        greedy_search_prob_normalized = greedy_search_prob/tf.reshape(tf.reduce_sum(greedy_search_prob,axis=1),[-1,1])
        greedy_embedding = tf.nn.embedding_lookup(embedding, greedy_search_sample)
        normalized_embedding = tf.multiply(tf.reshape(greedy_search_prob_normalized,[hps.batch_size,hps.k,1]), greedy_embedding)
        e2e_embedding = tf.reduce_mean(normalized_embedding,axis=1)
      else:
        e = []
        greedy_search_prob, greedy_search_sample = soft_top_k(alpha, output,
                                                              K=hps.k)  # (batch_size, k), (k, batch_size, vocab_size)
        greedy_search_prob_normalized = greedy_search_prob / tf.reshape(tf.reduce_sum(greedy_search_prob, axis=1),
                                                                        [-1, 1])

        for _ in range(hps.k):
          a_k = greedy_search_sample[_]
          e_k = tf.matmul(tf.reshape(greedy_search_prob_normalized[:,_],[-1,1]) * a_k, embedding)
          e.append(e_k)
        e2e_embedding = tf.reduce_sum(e, axis=0) # (batch_size, emb_dim)
      sampled_next_inputs = array_ops.gather_nd(e2e_embedding, where_sampling)
    else:
      if hps.hard_argmax:
        sampled_next_inputs = tf.nn.embedding_lookup(embedding, sample_ids_sampling)
      else: # using soft armax (greedy) proposed in: https://arxiv.org/abs/1704.06970
        #alpha_exp = tf.exp(alpha * (output_not_extended + G)) # (batch_size, vocab_size)
        #one_hot_scores = alpha_exp / tf.reduce_sum(alpha_exp, axis=1) #(batch_size, vocab_size)
        one_hot_scores = soft_argmax(alpha, output) #(batch_size, vocab_size)
        soft_argmax_embedding = tf.matmul(one_hot_scores, embedding) #(batch_size, emb_size)
        sampled_next_inputs = array_ops.gather_nd(soft_argmax_embedding, where_sampling)

    base_shape = array_ops.shape(inp)
    result1 = array_ops.scatter_nd(indices=where_sampling, updates=sampled_next_inputs, shape=base_shape)
    result2 = array_ops.scatter_nd(indices=where_not_sampling, updates=inputs_not_sampling, shape=base_shape)
    return result1 + result2

def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]
  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
  return res + bias_term
