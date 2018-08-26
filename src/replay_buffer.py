try:
  import queue
except:
  import Queue as queue
from random import shuffle
from random import seed
seed(123)
from threading import Thread
import numpy as np
import time
import tensorflow as tf
try:
  import Queue as Q  # ver. < 3.0
except ImportError:
  import queue as Q
from sklearn.preprocessing import normalize

PriorityQueue = Q.PriorityQueue


# Custom prioriy queue that is able to clear the queue once it is full
# and cut it to half. Therefore, everytime size of buffer is dqn_replay_buffer_size,
# we will keep half of the most valuable states and remove the rest to provide
# space for new experiences
class CustomQueue(PriorityQueue):
  '''
  A custom queue subclass that provides a :meth:`clear` method.
  '''
  def __init__(self, size):
    PriorityQueue.__init__(self, size)

  def clear(self):
    '''
    Clears all items from the queue.
    '''
    with self.mutex:
      unfinished = self.unfinished_tasks - len(self.queue)
      if unfinished <= 0:
        if unfinished < 0:
          raise ValueError('task_done() called too many times')
        self.all_tasks_done.notify_all()
      self.queue = self.queue[0:len(self.queue)/2]
      self.unfinished_tasks = unfinished + len(self.queue)
      self.not_full.notify_all()

  def isempty(self):
    with self.mutex:
      return len(self.queue) == 0

  def isfull(self):
    with self.mutex:
      return len(self.queue) == self.maxsize

class Transition(object):
  """
  A class for holding the experiences collected from seq2seq model
  """
  def __init__(self, state, action, state_prime, action_prime, reward, q_value, done):
    """
      Args:
        state: current decoder output state
        action: the greedy action selected from current decoder output
        state_prime: next decoder output state
        reward: reward of the greedy action selected
        q_value: Q-value of the greedy action selected
        done: whether we reached End-Of-Sequence or not
    """
    self.state = state # size: dqn_input_feature_len
    self.action = action # size: 1
    self.state_prime = state_prime # size: dqn_input_feature_len
    self.action_prime = action_prime
    self.reward = reward # size: vocab_size
    self.q_value = q_value # size: vocab_size
    self.done = done # true/false

  def __cmp__(self, item):
    """ PriorityQueue uses this functino to sort the rewards
      Args:
        We sort the queue such that items with higher rewards are in the head of max-heap
    """
    return cmp(item.reward, self.reward) # bigger numbers have more priority

class ReplayBatch(object):
  """ A class for creating batches required for training DDQN. """

  def __init__(self, hps, example_list, dqn_batch_size, use_state_prime = False, max_art_oovs = 0):
    """
      Args:
       hps: seq2seq model parameters
       example_list: list of experiences
       dqn_batch_size: DDQN batch size
       use_state_prime: whether to use the next decoder state to make the batch or the current one
       max_art_oovs: number of OOV tokens in current batch

      Properties:
        _x: The input to DDQN model for training, this is basically the decoder output (dqn_batch_size, dqn_input_feature_len)
        _y: The Q-estimation (dqn_batch_size, vocab_size)
        _y_extended: The Q-estimation (dqn_batch_size, vocab_size + max_art_oovs)
    """
    self._x = np.zeros((dqn_batch_size, hps.dqn_input_feature_len))
    self._y = np.zeros((dqn_batch_size, hps.vocab_size))
    self._y_extended = np.zeros((dqn_batch_size, hps.vocab_size + max_art_oovs))
    for i,e in enumerate(example_list):
      if use_state_prime:
        self._x[i,:]=e.state_prime
      else:
        self._x[i,:]=e.state
        self._y[i,:]=normalize([e.q_value[0:hps.vocab_size]], axis=1, norm='l1')
      if max_art_oovs == 0:
        self._y_extended[i,:] = normalize([e.q_value[0:hps.vocab_size]], axis=1, norm='l1')
      else:
        self._y_extended[i,:] = e.q_value

class ReplayBuffer(object):
  """ A class for implementing the priority experience buffer. """

  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  def __init__(self, hps):
    self._hps = hps
    self._buffer = CustomQueue(self._hps.dqn_replay_buffer_size)

    self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self._hps.dqn_batch_size)
    self._num_example_q_threads = 1 # num threads to fill example queue
    self._num_batch_q_threads = 1  # num threads to fill batch queue
    self._bucketing_cache_size = 100 # how many batches-worth of examples to load into cache before bucketing

    # Start the threads that load the queues
    self._example_q_threads = []
    for _ in range(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in range(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if they're dead
    self._watch_thread = Thread(target=self.watch_threads)
    self._watch_thread.daemon = True
    self._watch_thread.start()

  def next_batch(self):
    """Return a Batch from the batch queue.

    If mode='decode' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.

    Returns:
      batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
    """
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
      return None

    batch = self._batch_queue.get() # get the next Batch
    return batch

  @staticmethod
  def create_batch(_hps, batch, batch_size, use_state_prime=False, max_art_oovs=0):
    """ Create a DDQN-compatible batch from the input transitions

      Args:
        _hps: seq2seq model parameters
        batch: a list of Transitions
        dqn_batch_size: DDQN batch size
        use_state_prime: whether to use the next decoder state to make the batch or the current one
        max_art_oovs: number of OOV tokens in current batch

      Returns:
        An object of ReplayBatch class
    """

    return ReplayBatch(_hps, batch, batch_size, use_state_prime, max_art_oovs)

  def fill_example_queue(self):
    """Reads data from file and processes into Examples which are then placed into the example queue."""
    while True:
      try:
        input_gen = self._example_generator().next()
      except StopIteration: # if there are no more examples:
        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        raise Exception("single_pass mode is off but the example generator is out of data; error.")
      self._example_queue.put(input_gen) # place the pair in the example queue.

  def fill_batch_queue(self):
    """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue."""
    while True:
      # Get bucketing_cache_size-many batches of Examples into a list, then sort
      inputs = []
      for _ in range(self._hps.dqn_batch_size * self._bucketing_cache_size):
        inputs.append(self._example_queue.get())

      # feed back all the samples to the buffer
      self.add(inputs)

      # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
      batches = []
      for i in range(0, len(inputs), self._hps.dqn_batch_size):
        batches.append(inputs[i:i + self._hps.dqn_batch_size])
        shuffle(batches)
      for b in batches:  # each b is a list of Example objects
        self._batch_queue.put(ReplayBatch(self._hps, b, self._hps.dqn_batch_size))

  def watch_threads(self):
    """Watch example queue and batch queue threads and restart if dead."""
    while True:
      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()

  def add(self, items):
    """ Adding a list of experiences to the buffer. When buffer is full,
      we get rid of half of the least important experiences and keep the rest.

      Args:
        items: A list of experiences of size (batch_size, k, max_dec_steps, hidden_dim)
    """
    for item in items:
      if not self._buffer.isfull():
        self._buffer.put_nowait(item)
      else:
        print('Replay Buffer is full, getting rid of unimportant transitions...')
        self._buffer.clear()
        self._buffer.put_nowait(item)
    print('ReplayBatch size: {}'.format(self._buffer.qsize()))
    print('ReplayBatch example queue size: {}'.format(self._example_queue.qsize()))
    print('ReplayBatch batch queue size: {}'.format(self._batch_queue.qsize()))

  def _buffer_len(self):
    return self._buffer.qsize()

  def _example_generator(self):
    while True:
      if not self._buffer.isempty():
        item = self._buffer.get_nowait()
        self._buffer.task_done()
        yield item
