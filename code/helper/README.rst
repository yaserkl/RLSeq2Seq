********************
Helper Codes
********************

=================
CNN-Daily Mail
=================

-------------------------------------------
First Option: Download the processed data
-------------------------------------------

`@JafferWilson <https://github.com/JafferWilson/>`_ provided the processed data, which you can download from `here <https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail>`_.

----------------------------------------------------------------------------------
Second Option: Process dataset using `@abisee <https://github.com/abisee/>`_ code
----------------------------------------------------------------------------------
Check out the pre-processing in `here <https://github.com/abisee/cnn-dailymail>`_.

-------------------------------------------------------
Third Option: Process dataset using our helper code
-------------------------------------------------------
----------------------
Download Raw Data
----------------------
Through our helper codes, we first download all the articles in both CNN and Daily Mail dataset along with the title of each article and their highlights. All you need to do is to download the "*.story" files for each dataset from `here <http://cs.nyu.edu/~kcho/DMQA/>`_. As per @abisee, these files contain 114 examples for which the article text is missing, we store list of these files in the filter_data.txt file and during our pre-processing we remove them from the dataset.

------------------------------------------------------------------
Download article, headline, and highlights from raw data
------------------------------------------------------------------
Then, run following to start downloading the articles and headline of each article. Please note that, this dataset doesn't have headline information, therefore we have to download the headline from each url by re-downloading the whole article.

.. code:: bash

    python cnn_dm_downloader.py ~/data/cnn_dm/[cnn,dailymail]/ ~/data/cnn_dm/[cnn,dailymail]/processed/ article

And run this to download the highlights:

.. code:: bash

    python cnn_dm_downloader.py ~/data/cnn_dm/[cnn,dailymail]/ ~/data/cnn_dm/[cnn,dailymail]/processed/ highlight

----------------------
Pre-process the data
----------------------

The main pre-processing that we do on this dataset is the following:

1. Use `Spacy <http://spacy.io/>`_ to tokenize and process the articles and its highlights.

2. We first sentence tokenize each article and then use word tokenization and re-join everything to get the full sentence.

3. Each sentences is terminated by an "EOS" token rather than "." and each article (highlight, headline) has the following format:

  "<d> <s> first sentence ... </s> <s> second sentence ... </s> </d>".

4. Replacing all Named-Entities with "_" delimited version of them. For instance, given "Virgini Tech", we get "virginia_tech"
5. Collect POS tag and Named-Entities of each token in the text and store them in a separate file.

We can run our pre-processing using the following command:

.. code:: bash

    python cnn_dm_data_maker.py ~/data/cnn_dm/[cnn,dailymail]/ ~/data/cnn_dm/[cnn,dailymail]/[article,title,highlight] [article,title/highlight]

-------------------------
Create Train, Dev, Test
-------------------------

Finally, we use the following to create our train, dev, test datasets:

.. code:: bash

    python cnn_dm_data_merger.py ~/data/cnn_dm/ cnn_dm.txt

=================
Newsroom Dataset
=================

----------------------
Download Raw Data
----------------------

You can follow this `link <https://github.com/clic-lab/newsroom>`_ to download and extract this dataset. After downloading you should have "[train,test,dev].data" files.

----------------------
Run Pre-processing
----------------------

This is completely similar to the CNN/Daily Mail pre-processing. The input directory must contain "[train,test,dev].data" files.

.. code:: bash

    python summaries_data_maker.py ~/data/newsroom/

