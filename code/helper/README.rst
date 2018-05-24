
********************
CNN-Daily Mail
********************
Through our helper codes, we first download all the articles in both CNN and Daily Mail dataset along with the title of each article and their highlights. All you need to do is to download the *.story files for each dataset from the following links:

Then run this to start downloading the articles and headline of each article:
.. code:: bash

    python cnn_dm_downloader.py ~/data/cnn_dm/cnn/ ~/data/cnn_dm/cnn/processed/ article

And run this to download the highlights:
.. code:: bash

    python cnn_dm_downloader.py ~/data/cnn_dm/cnn/ ~/data/cnn_dm/cnn/processed/ highlight





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

.. image:: docs/_img/seq2seqmodel.png
    :target: docs/_img/seq2seqmodel.png


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
Please refer to this link to access them:

code/helper

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

  - Version?

~~~~~~~~~~~~~~~~~~~
GPU
~~~~~~~~~~~~~~~~~~~

  - Cuda version?
  - Cudnn?



-----------------
Training
-----------------

-----------------
Evaluation
-----------------

~~~~~~~~~~~~~~~~~~~
Evaluation metrics
~~~~~~~~~~~~~~~~~~~



---------------------------------------------------------------------------

===============
Citation
===============



---------------------------------------------------------------------------

===============
Aknowledgement
===============
