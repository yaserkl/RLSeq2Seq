
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

The goal of this repository is ...


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


============
Motivation
============

In recent years, text summarization moved from traditional bag of word models to more advanced methods based on Recurrent Neural Networks (RNN). The underlying framework of all these models are usually a deep neural network which contains an encoder module and a decoder module. The encoder processes the input data and a decoder receive the output of the encoder and generates the final output. Although simply using an encoder/decoder framework would, most of the time, produce better results than traditional methods on text summarization, researchers proposed additional improvements over these models by using attention-based models, pointer-generation models, and self-attention models. However, all these models suffer from a common problem known as exposure bias. In this work, we first study various solutions suggested for avoiding exposure bias and show how these solutions perform on abstractive text summarization and finally propose our solution, SoftE2E, that reaches state-of-the-art result on CNN/Daily Mail dataset.

.. image:: docs/_img/seq2seq.png
    :target: docs/_img/seq2seq.png

---------------------------------------------------------------------------

====================
Requirements
====================
-------------
Python
-------------

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
in the summarization results, therefore, we strongly suggest to use these pre-processed files for
all the trainings.

---------------------------------------------------------------------------

====================
Running Experiments
====================
This code is a general framework for a variety of different modes that supporst the following features:

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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Scheduled Sampling using Hard-Argmax and Greedy selection (`Bengio et al <https://arxiv.org/abs/1506.03099>`_.):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=scheduled-sampling-hardargmax-greedy --batch_size=80 --max_iter=43083 --scheduled_sampling=True --sampling_probability=9.28421E-06 --hard_argmax=True --greedy_scheduled_sampling=True

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Scheduled Sampling using Soft-Argmax and Sampling selection (`Goyal et al <https://arxiv.org/abs/1506.03099>`_.):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=scheduled-sampling-softargmax-sampling --batch_size=80 --max_iter=43083 --scheduled_sampling=True --sampling_probability=9.28421E-06 --hard_argmax=False --greedy_scheduled_sampling=False --alpha=10


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End2EndBackProp (`Ranzato et al <https://arxiv.org/abs/1511.06732>`_.):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=scheduled-sampling-end2endbackprop --batch_size=80 --max_iter=43083 --scheduled_sampling=True --sampling_probability=9.28421E-06 --hard_argmax=True --E2EBackProp=True --k=4

---------------------------------------------------------------------------

-------------------------------------------------------------------------------------------
Policy-Gradient w. Self-Critic learning and temporal attention and intra-decoder attention
-------------------------------------------------------------------------------------------

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
Here, we use a different GPU for evalation, but we can use the same GPU if we decrease the number of batches. In our implementation, we use a batch size of 8 for evaluation but for each eval step, we iterate over the validation dataset 100 times. This is similar to finding the evaluation error on a batch size of 800. This will help to decrease the memory required by the evaluation process and provide options for running both training and eval on one GPU.

.. code:: bash

    CUDA_VISIBLE_DEVICES=1 python code/run_summarization.py --mode=eval --data_path=$HOME/data/cnn_dm/finished_files/chunked/val_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --batch_size=8 --use_temporal_attention=True --intradecoder=True --rl_training=False


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Activating MLE+RL training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As suggested by `Paulus et al <https://arxiv.org/abs/1705.04304>`_, we use a linear transition from Cross-Entropy loss to RL loss so that in the end we completely rely on RL loss to train the model. The parameter eta controls this transition. We set eta to be eta = 1/(max RL iteration).

First, add required training parameter to the model:

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --batch_size=80 --max_iter=40000 --intradecoder=True --use_temporal_attention=True --eta=2.17599E-05 --rl_training=True --convert_to_reinforce_model=True


Then, start running the model with MLE+RL training loss: 

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --batch_size=80 --max_iter=40000 --intradecoder=True --use_temporal_attention=True --eta=2.17599E-05 --rl_training=True

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

    CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=decode --data_path=$HOME/data/cnn_dm/finished_files/chunked/test_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --rl_training=True --intradecoder=True --use_temporal_attention=True --single_pass=1 --beam_size=4 --decode_after=0


---------------------------------------------------------------------------


----------------------------------------------------
Actor-Critic model through DDQN and Dueling network
----------------------------------------------------
The general framework for Actor-Critic model is as follows:

.. image:: docs/_img/rlseq.png
    :target: docs/_img/rlseq.png

In our impolementaion the Actor is the pointer-generator model and the Critic is a regression model that minimizes the Q-value estimation using Double Deep Q Network (DDQN). The code is implemented such that the DDQN training is on a different thread from the main thread and we collect experiences for this network asynchronously from the Actor model. Therefore, for each batch we collect (batch_size * max_dec_steps) states for the DDQN training. We implemented the `prioritized replay buffer <https://arxiv.org/abs/1511.05952>`_. and during DDQN training we always select our minibatches such that they contain experiences that have the best partial reward according to the ground-truth summary. We added an option of training DDQN based on true Q-estimation and offered a scheduled-sampling process for training this network. Please note that, training DDQN using true Q-estimation will significantly reduce the speed of training, due to collection of true Q-values. Therefore, we suggest to only activate this for a few iterations. As suggested by `Bahdanau et al <https://arxiv.org/pdf/1607.07086.pdf>`_. it is also good to use a fixed pre-trained Actor to pre-train the Critic model first and then start training both models, simultaneously. For instance, we can use the following set of codes to run a similar experience as `Bahdanau et al <https://arxiv.org/pdf/1607.07086.pdf>`_.:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-Training the Actor using only MLE loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=actor-critic-ddqn --batch_size=80 --max_iter=20000

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Adding Critic model to the current model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We can use Dueling network to train the DDQN by activating dueling_net flag. Moreover, we can choose to update the target network using polyak averaging by dqn_polyak_averaging flag.

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=actor-critic-ddqn --batch_size=80 --max_iter=21000 --convert_to_reinforce_model=True --ac_training=True --dueling_net=True --dqn_polyak_averaging=True


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Start Pre-Training Critic with fixed Actor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use dqn_pretrain_steps flag to set how many iteration you want to pre-train the Critic.

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=actor-critic-ddqn --batch_size=80 --ac_training=True --dqn_pretrain=True --dueling_net=True --dqn_polyak_averaging=True


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Start Training Actor/Critic using True Q-Estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We can run Actor in one GPU and Critic in another GPU simply by using a different GPU number for Critic using dqn_gpu_num option. Also as mentioned before, we should avoid using true Q-estimation for long, therefore, we use true estimation to train DDQN for only 1000 iterations.

.. code:: bash

    CUDA_VISIBLE_DEVICES=0,1 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=actor-critic-ddqn --batch_size=80 --max_iter=22000 --ac_training=True --dueling_net=True --dqn_polyak_averaging=True --calculate_true_q=True --dqn_gpu_num=1

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Start Training Actor/Critic using Q-Estimates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We can run Actor in one GPU and Critic in another GPU simply by using a different GPU number for Critic using dqn_gpu_num option.

.. code:: bash

    CUDA_VISIBLE_DEVICES=0,1 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=actor-critic-ddqn --batch_size=80 --max_iter=40000 --ac_training=True --dueling_net=True --dqn_polyak_averaging=True --dqn_gpu_num=1

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Decoding based on Actor and Critic estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=decode --data_path=$HOME/data/cnn_dm/finished_files/chunked/test_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=actor-critic-ddqn --ac_training=True --dueling_net=True --dqn_polyak_averaging=True --dqn_gpu_num=1 --single_pass=1 --beam_size=4


===============
Citation
===============



---------------------------------------------------------------------------

===============
Aknowledgement
===============
Thanks `@atorfi <https://github.com/atorfi/>`_ for his help on preparing this documentation.