
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

In recent years, text summarization moved from traditional bag of word models to more
advanced methods based on Recurrent Neural Networks (RNN). The underlying framework of all these models are usually a deep neural network which contains an encoder
module and a decoder module. The encoder processes the input data and a decoder receive the output of the encoder and gener-
ates the final output. Although simply using an encoder/decoder framework would, most
of the time, produce better results than traditional methods on text summarization, researchers proposed additional improvements over these models by using attention-based
models, pointer-generation models, and self-attention models. However, all these models
suffer from a common problem known as exposure bias. In this work, we first study various solutions suggested for avoiding exposure
bias and show how these solutions perform on abstractive text summarization and finally propose our solution, SoftE2E, that reaches state-
of-the-art result on CNN/Daily Mail dataset.

.. image:: docs/_img/seq2seq.png
    :target: docs/_img/seq2seq.png

.. image:: docs/_img/rlseq.png
    :target: docs/_img/rlseq.png

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
Code Implementation
====================

-----------------
Dependencies
-----------------

~~~~~~~~~~~~~~~~~~~
Python
~~~~~~~~~~~~~~~~~~~

Python requirements can be installed as follows:

.. code:: bash

    pip install -r python_requirements.txt

~~~~~~~~~~~~~~~~~~~
TensorFlow
~~~~~~~~~~~~~~~~~~~

  - Use Tensorflow 1.4

~~~~~~~~~~~~~~~~~~~
GPU
~~~~~~~~~~~~~~~~~~~

  - CUDA 8
  - CUDNN 6

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
Policy-Gradient w. Self-Critic learning and temporal attention and intra-decoder attention
-------------------------------------------------------------------------------------------

`Paulus et al <https://arxiv.org/abs/1705.04304>`_, proposed a self-critic policy-gradient model for abstractive text summarization. The following figure represents how this method works and how we implemented this method:

.. image:: docs/_img/selfcritic.png
    :target: docs/_img/selfcritic.png

To replicate their experiment, we can use the following set of processes:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-Training using only MLE loss with intradecoder attention and temporal attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python code/run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --batch_size=80 --max_iter=17951 --use_temporal_attention=True --intradecoder=True --rl_training=False


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

    CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --batch_size=80 --max_iter=44000 --intradecoder=True --use_temporal_attention=True --eta=2.17599E-05 --rl_training=True --convert_to_reinforce_model=True


Then, start running the model with MLE+RL training loss: 

.. code:: bash

    CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=train --data_path=$HOME/data/cnn_dm/finished_files/chunked/train_* --vocab_path=$HOME/data/cnn_dm/finished_files/vocab --log_root=$HOME/working_dir/cnn_dm/RLSeq2Seq/ --exp_name=intradecoder-temporalattention-withpretraining --batch_size=80 --max_iter=44000 --intradecoder=True --use_temporal_attention=True --eta=2.17599E-05 --rl_training=True

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

===============
Citation
===============



---------------------------------------------------------------------------

===============
Aknowledgement
===============
Thanks `@atorfi <https://github.com/atorfi/>`_ for his help on preparing this documentation.