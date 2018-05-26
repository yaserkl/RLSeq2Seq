
********************
RLSeq2Seq
********************

.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/yaserkl/RLSeq2Seq/pulls
.. image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
      :target: https://www.python.org/
.. image:: https://img.shields.io/pypi/l/ansicolortags.svg
      :target: https://github.com/yaserkl/RLSeq2Seq/blob/master/LICENSE.txt
.. image:: https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg
      :target: https://github.com/yaserkl/RLSeq2Seq/graphs/contributors
.. image:: https://img.shields.io/github/issues/Naereen/StrapDown.js.svg
      :target: https://github.com/yaserkl/RLSeq2Seq/issues
.. image:: https://img.shields.io/badge/arXiv-1805.09461-red.svg?style=flat
   :target: https://arxiv.org/abs/1805.09461

NOTE: THE CODE IS UNDER DEVELOPMENT, PLEASE ALWAYS PULL THE LATEST VERSION FROM HERE.

This repository contains the code developed in TensorFlow_ for the following paper:


| `Deep Reinforcement Learning For Sequence to Sequence Models`_,
| by: `Yaser Keneshloo`_, `Tian Shi`_, `Naren Ramakrishnan`_, and `Chandan K. Reddy`_


.. _Deep Reinforcement Learning For Sequence to Sequence Models: https://arxiv.org/abs/1805.09461
.. _TensorFlow: https://www.tensorflow.org/
.. _Yaser Keneshloo: https://github.com/yaserkl
.. _Tian Shi: http://life-tp.com/Tian_Shi/
.. _Chandan K. Reddy: http://people.cs.vt.edu/~reddy/
.. _Naren Ramakrishnan: http://people.cs.vt.edu/naren/


If you used this code, please kindly consider citing the following paper:

.. code:: shell

    @article{keneshloo2018deep,
     title={Deep Reinforcement Learning For Sequence to Sequence Models},
     author={Keneshloo, Yaser and Shi, Tian and Ramakrishnan, Naren and Reddy, Chandan K.},
     journal={arXiv preprint arXiv:1805.09461},
     year={2018}
    }



#################
Table of Contents
#################
.. contents::
  :local:
  :depth: 3


..  Chapter 1 Title
..  ===============

..  Section 1.1 Title
..  -----------------

..  Subsection 1.1.1 Title
..  ~~~~~~~~~~~~~~~~~~~~~~

.. image:: docs/_img/rlseq.png
    :target: docs/_img/rlseq.png

============
Motivation
============

In recent years, sequence-to-sequence (seq2seq) models are used in a variety of tasks from machine translation, headline generation, text summarization, speech to text, to image caption generation. The underlying framework of all these models are usually a deep neural network which contains an encoder and decoder. The encoder processes the input data and a decoder receives the output of the encoder and generates the final output. Although simply using an encoder/decoder model would, most of the time, produce better result than traditional methods on the above-mentioned tasks, researchers proposed additional improvements over these sequence to sequence models, like using an attention-based model over the input, pointer-generation models, and self-attention models. However, all these seq2seq models suffer from two common problems: 1) exposure bias and 2) inconsistency between train/test measurement. Recently a completely fresh point of view emerged in solving these two problems in seq2seq models by using methods in Reinforcement Learning (RL). In these new researches, we try to look at the seq2seq problems from the RL point of view and we try to come up with a formulation that could combine the power of RL methods in decision-making and sequence to sequence models in remembering long memories. In this paper, we will summarize some of the most recent frameworks that combines concepts from RL world to the deep neural network area and explain how these two areas could benefit from each other in solving complex seq2seq tasks. In the end, we will provide insights on some of the problems of the current existing models and how we can improve
them with better RL models. We also provide the source code for implementing most of the models that will be discussed in this paper on the complex task of abstractive text summarization.

---------------------------------------------------------------------------

====================
Requirements
====================
-------------
Python
-------------
  - Use Python 2.7

Python requirements can be installed as follows:

.. code:: bash

    pip install -r python_requirements.txt

-------------
TensorFlow
-------------

  - Use Tensorflow 1.4

-------------
GPU
-------------

  - CUDA 8
  - CUDNN 6

---------------------------------------------------------------------------

============
DATASET
============
----------------------
CNN/Daily Mail dataset
----------------------
https://github.com/abisee/cnn-dailymail

----------------------
Newsroom dataset
----------------------
https://summari.es/

We have provided helper codes to download the cnn-dailymail dataset and
pre-process this dataset and newsroom dataset.
Please refer to `this link <code/helper>`_ to access them.

We saw a large improvement on the ROUGE measure by using our processed version of these datasets
in the summarization results, therefore, we strongly suggest using these pre-processed files for
all the training.

---------------------------------------------------------------------------

====================
Running Experiments
====================
This code is a general framework for a variety of different modes that supports the following features:

1. Scheduled Sampling, Soft-Scheduled Sampling, and End2EndBackProp.
2. Policy-Gradient w. Self-Critic learning and temporal attention and intra-decoder attention:

   A. Following `A Deep Reinforced Model for Abstractive Summarization <https://arxiv.org/abs/1705.04304>`_
3. Actor-Critic model through DDQN and Dueling network based on these papers:

   A. `Deep Reinforcement Learning with Double Qlearning <https://arxiv.org/abs/1509.06461>`_
   B. `Dueling Network Architectures for Deep Reinforcement Learning <https://arxiv.org/abs/1511.06581>`_
   C. `An ActorCritic Algorithm for Sequence Prediction <https://arxiv.org/abs/1607.07086>`_



---------------------------------------------------------------------------

-------------------------------------------------------------------------------------------
Scheduled Sampling, Soft-Scheduled Sampling, and End2EndBackProp
-------------------------------------------------------------------------------------------
`Bengio et al <https://arxiv.org/abs/1506.03099>`_. proposed the idea of scheduled sampling for avoiding exposure bias problem. Recently, `Goyal et al <https://arxiv.org/abs/1506.03099>`_. proposed a differentiable relaxtion of this method, by using soft-argmax rather hard-argmax, that solves the back-propagation error that exists in this model. Also, `Ranzato et al <https://arxiv.org/abs/1511.06732>`_. proposed another simple model called End2EndBackProp for avoiding exposure bias problem. To train a model based on each of these papers, we provide different flags as follows:

 +----------------------------+---------+-------------------------------------------------------------------+
 | Parameter                  | Default | Description                                                       |
 +============================+=========+===================================================================+
 | scheduled_sampling         |  False  | whether to do scheduled sampling or not                           |
 +----------------------------+---------+-------------------------------------------------------------------+
 | sampling_probability       |    0    | epsilon value for choosing ground-truth or model output           |
 +----------------------------+---------+-------------------------------------------------------------------+
 | fixed_sampling_probability |  False  | Whether to use fixed sampling probability or adaptive             |
 +----------------------------+---------+-------------------------------------------------------------------+
 | hard_argmax                |  True   | Whether to use soft argmax or hard argmax                         |
 +----------------------------+---------+-------------------------------------------------------------------+
 | greedy_scheduled_sampling  |  False  | Whether to use greedy or sample for the output, True means greedy |
 +----------------------------+---------+-------------------------------------------------------------------+
 | E2EBackProp                |  False  | Whether to use E2EBackProp algorithm to solve exposure bias       |
 +----------------------------+---------+-------------------------------------------------------------------+
 | alpha                      |    1    | soft argmax argument                                              |
 +----------------------------+---------+-------------------------------------------------------------------+


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Scheduled Sampling using Hard-Argmax and Greedy selection (`Bengio et al <https://arxiv.org/abs/1506.03099>`_.):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=scheduled-sampling-hardargmax-greedy --batch_size=80 --max_iter=40000 --scheduled_sampling=True --sampling_probability=2.5E-05 --hard_argmax=True --greedy_scheduled_sampling=True

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Scheduled Sampling using Soft-Argmax and Sampling selection (`Goyal et al <https://arxiv.org/abs/1506.03099>`_.):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=scheduled-sampling-softargmax-sampling --batch_size=80 --max_iter=40000 --scheduled_sampling=True --sampling_probability=2.5E-05 --hard_argmax=False --greedy_scheduled_sampling=False --alpha=10


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End2EndBackProp (`Ranzato et al <https://arxiv.org/abs/1511.06732>`_.):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=scheduled-sampling-end2endbackprop --batch_size=80 --max_iter=40000 --scheduled_sampling=True --sampling_probability=2.5E-05 --hard_argmax=True --E2EBackProp=True --k=4

---------------------------------------------------------------------------

-------------------------------------------------------------------------------------------
Policy-Gradient w. Self-Critic learning and temporal attention and intra-decoder attention
-------------------------------------------------------------------------------------------

 +----------------------------+-----------------+---------------------------------------------------------------------+
 | Parameter                  |     Default     | Description                                                         |
 +============================+=================+=====================================================================+
 | rl_training                |      False      | Start policy-gradient training                                      |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 |                            |                 | Convert a pointer model to a reinforce model.                       |
 |                            |                 | Turn this on and run in train mode. Your current training model     |
 | convert_to_reinforce_model |      False      | will be copied to a new version (same name with _cov_init appended) |
 |                            |                 | that will be ready to run with coverage flag turned on,             |
 |                            |                 | for the coverage training stage.                                    |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | intradecoder               |      False      | Use intradecoder attention or not                                   |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | use_temporal_attention     |      True       | Whether to use temporal attention or not                            |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | matrix_attention           |      False      | Use matrix attention, Eq. 2 in https://arxiv.org/pdf/1705.04304.pdf |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | eta                        |        0        | RL/MLE scaling factor, 1 means use RL loss, 0 means use MLE loss    |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | fixed_eta                  |      False      | Use fixed value for eta or adaptive based on global step            |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | gamma                      |       0.99      | RL reward discount factor                                           |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | reward_function            | rouge_l/f_score | Either bleu or one of the rouge measures                            |
 |                            |                 | (rouge_1/f_score, rouge_2/f_score,rouge_l/f_score)                  |
 +----------------------------+-----------------+---------------------------------------------------------------------+

`Paulus et al <https://arxiv.org/abs/1705.04304>`_. proposed a self-critic policy-gradient model for abstractive text summarization. The following figure represents how this method works and how we implemented this method:

.. image:: docs/_img/selfcritic.png
    :target: docs/_img/selfcritic.png

To replicate their experiment, we can use the following set of processes:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-Training using only MLE loss with intradecoder attention and temporal attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --batch_size=80 --max_iter=20000 --use_temporal_attention=True --intradecoder=True --rl_training=False


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluation the pre-trained model on validation data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here, we use a different GPU for evaluation, but we can use the same GPU if we decrease the number of batches. In our implementation, we use a batch size of 8 for evaluation but for each eval step, we iterate over the validation dataset 100 times. This is similar to finding the evaluation error on a batch size of 800. This will help to decrease the memory required by the evaluation process and provide options for running both training and eval on one GPU.

.. code:: bash

    CUDA_VISIBLE_DEVICES=1 python code/run_summarization.py --mode=eval --data_path=$HOME/data/cnn_dm/finished_files/chunked/val_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --batch_size=8 --use_temporal_attention=True --intradecoder=True --rl_training=False


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Activating MLE+RL training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As suggested by `Paulus et al <https://arxiv.org/abs/1705.04304>`_, we use a linear transition from Cross-Entropy loss to RL loss so that in the end we completely rely on RL loss to train the model. The parameter eta controls this transition. We set eta to be eta = 1/(max RL iteration).

First, add required training parameter to the model:

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --batch_size=80 --max_iter=40000 --intradecoder=True --use_temporal_attention=True --eta=2.5E-05 --rl_training=True --convert_to_reinforce_model=True


Then, start running the model with MLE+RL training loss:

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --batch_size=80 --max_iter=40000 --intradecoder=True --use_temporal_attention=True --eta=2.5E-05 --rl_training=True

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluating MLE+RL training on validation data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    CUDA_VISIBLE_DEVICES=1 python code/run_summarization.py --mode=eval --data_path=$HOME/data/cnn_dm/finished_files/chunked/val_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --batch_size=8 --use_temporal_attention=True --intradecoder=True --rl_training=True

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Start decoding the trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We use ROUGE as the evaluation metrics.

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=decode --data_path=$HOME/data/cnn_dm/finished_files/chunked/test_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --rl_training=True --intradecoder=True --use_temporal_attention=True --single_pass=1 --beam_size=4 --decode_after=0

---------------------------------------------------------------------------

----------------------------------------------------
Actor-Critic model through DDQN and Dueling network
----------------------------------------------------

 +----------------------------+-----------------+---------------------------------------------------------------------+
 | Parameter                  |     Default     | Description                                                         |
 +============================+=================+=====================================================================+
 | ac_training                |      False      | Use Actor-Critic learning by DDQN.                                  |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | dqn_scheduled_sampling     |      False      | Whether to use scheduled sampling to use estimates of DDQN model    |
 |                            |                 | vs the actual Q-estimates values                                    |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | dqn_layers                 |   512,256,128   | DDQN dense hidden layer size.                                       |
 |                            |                 | It will create three dense layers with 512, 256, and 128 size       |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | dqn_replay_buffer_size     |     100000      | Size of the replay buffer                                           |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | dqn_batch_size             |       100       | Batch size for training the DDQN model                              |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | dqn_target_update          |      10000      | Update target Q network every 10000 steps                           |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | dqn_sleep_time             |        2        | Train DDQN model every 2 seconds                                    |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | dqn_gpu_num                |        1        | GPU number to train the DDQN                                        |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | dueling_net                |       True      | Whether to use Duelling Network to train the model                  |
 |                            |                 | https://arxiv.org/pdf/1511.06581.pdf                                |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | dqn_polyak_averaging       |       True      | Whether to use Polyak averaging to update the target Q network      |
 |                            |                 | parameters: Psi^{\prime} = (tau * Psi^{\prime})+ (1-tau)*Psi        |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | calculate_true_q           |      False      | Whether to use true Q-values to train DDQN                          |
 |                            |                 | or use DDQN's estimates to train it                                 |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | dqn_pretrain               |      False      | Pretrain the DDQN network with fixed Actor model                    |
 +----------------------------+-----------------+---------------------------------------------------------------------+
 | dqn_pretrain_steps         |      10000      | Number of steps to pre-train the DDQN                               |
 +----------------------------+-----------------+---------------------------------------------------------------------+

The general framework for the Actor-Critic model is as follows:

.. image:: docs/_img/rlseq.png
    :target: docs/_img/rlseq.png

In our implementation, the Actor is the pointer-generator model and the Critic is a regression model that minimizes the Q-value estimation using Double Deep Q Network (DDQN). The code is implemented such that the DDQN training is on a different thread from the main thread and we collect experiences for this network asynchronously from the Actor model. Therefore, for each batch, we collect (batch_size * max_dec_steps) states for the DDQN training. We implemented the `prioritized replay buffer <https://arxiv.org/abs/1511.05952>`_. and during DDQN training we always select our mini batches such that they contain experiences that have the best partial reward according to the ground-truth summary. We added an option of training DDQN based on true Q-estimation and offered a scheduled-sampling process for training this network. Please note that training DDQN using true Q-estimation will significantly reduce the speed of training, due to the collection of true Q-values. Therefore, we suggest to only activate this for a few iterations. As suggested by `Bahdanau et al <https://arxiv.org/pdf/1607.07086.pdf>`_. it is also good to use a fixed pre-trained Actor to pre-train the Critic model first and then start training both models, simultaneously. For instance, we can use the following set of codes to run a similar experiment as `Bahdanau et al <https://arxiv.org/pdf/1607.07086.pdf>`_.:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-Training the Actor using only MLE loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=actor-critic-ddqn --batch_size=80 --max_iter=20000

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adding Critic model to the current model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We can use Dueling network to train the DDQN by activating ``dueling_net`` flag. Moreover, we can choose to update the target network using polyak averaging by ``dqn_polyak_averaging`` flag.

.. code:: bash

    CUDA_VISIBLE_DEVICES=0,1 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=actor-critic-ddqn --batch_size=80 --max_iter=21000 --ac_training=True --dueling_net=True --dqn_polyak_averaging=True --convert_to_reinforce_model=True --dqn_gpu_num=1


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Start Pre-Training Critic with fixed Actor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use ``dqn_pretrain_steps`` flag to set how many iteration you want to pre-train the Critic.

.. code:: bash

    CUDA_VISIBLE_DEVICES=0,1 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=actor-critic-ddqn --batch_size=80 --ac_training=True --dqn_pretrain=True --dueling_net=True --dqn_polyak_averaging=True --dqn_gpu_num=1


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Start Training Actor/Critic using True Q-Estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We can run Actor in one GPU and Critic in another GPU simply by using a different GPU number for Critic using ``dqn_gpu_num`` option. Also as mentioned before, we should avoid using true Q-estimation for long, therefore, we use true estimation to train DDQN for only 1000 iterations.

.. code:: bash

    CUDA_VISIBLE_DEVICES=0,1 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=actor-critic-ddqn --batch_size=80 --max_iter=22000 --ac_training=True --dueling_net=True --dqn_polyak_averaging=True --calculate_true_q=True --dqn_gpu_num=1

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Start Training Actor/Critic using Q-Estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please note that we don't use ``calculate_true_q`` flag, anymore.

.. code:: bash

    CUDA_VISIBLE_DEVICES=0,1 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=actor-critic-ddqn --batch_size=80 --max_iter=40000 --ac_training=True --dueling_net=True --dqn_polyak_averaging=True --dqn_gpu_num=1

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Decoding based on Actor and Critic estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=decode --data_path=$HOME/data/cnn_dm/finished_files/chunked/test_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=actor-critic-ddqn --ac_training=True --dueling_net=True --dqn_polyak_averaging=True --dqn_gpu_num=1 --single_pass=1 --beam_size=4


---------------------------------------------------------------------------

Please note that we can use options such as ``intradecoder``, ``temporal_attention``, ``E2EBackProp``, ``scheduled_sampling``, etc in Actor-Critic models, too. Using these options will help to have a better performing Actor model.

.. .. code:: bash

==================
Under Development
==================

~~~~~~~~~~~~~~~~~~~~~~~~
Supporting YAML configs
~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Storing ReplayBuffer in disk for Actor-Critic models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

===============
Aknowledgement
===============
Thanks `@astorfi <https://github.com/astorfi/>`_ for his help on preparing this documentation.
