# This code will tokenize and detect Named-Entities and concat them with "_".
# Therefore, given the following sentence:
# --President Trump said he based his decision on "open hostility" from North Korea.--
# it will return:
# --president_trump said he based his decision on " open hostility " from north_korea EOS--

# python cnn_dm_data_merger.py ~/data/cnn_dm/ cnn_dm.txt ./filter_files.txt

# ~/data/cnn_dm/: This directory must contains the following subdirectories:
# --~/data/cnn_dm/cnn/
# ----~/data/cnn_dm/cnn/[article,title,highlight]_spacy_line
# ----~/data/cnn_dm/cnn/[article,title,highlight]_spacy_pos
# ----~/data/cnn_dm/cnn/[article,title,highlight]_spacy_ner
# --~/data/cnn_dm/dailymail/
# ----~/data/cnn_dm/dailymail/[article,title,highlight]_spacy_line
# ----~/data/cnn_dm/dailymail/[article,title,highlight]_spacy_pos
# ----~/data/cnn_dm/dailymail/[article,title,highlight]_spacy_ner

from glob import glob
import os, sys
from collections import defaultdict
import pandas as pd
import numpy as np
from collections import Counter
from tensorflow.core.example import example_pb2
import struct

root_dir = sys.argv[1]
outfile = sys.argv[2]
filter_file = sys.argv[3]

datasets = ['cnn','dailymail']
df = defaultdict(list)
fw = open('{}/{}'.format(root_dir,outfile),'w')
for dataset in datasets:
    working_dir = os.path.join(root_dir, dataset)
    files = glob('{}/article_spacy_line/*'.format(working_dir))
    print(len(files))
    filter_files = [k.strip() for k in open(filter_file).readlines()]
    files = [k for k in files if k.split('/')[-1] not in filter_files]
    print(len(files))
    for fl in files:
        filename = fl.split('/')[-1]
        fasl = open('{}/article_spacy_line/{}'.format(working_dir,filename))
        lines = ['<s> {} </s> '.format(' '.join(k.strip().split())) for k in fasl]
        if len(lines)==0:
            continue
        df['article'].append(''.join(lines))
        article_line = '<d> {}</d>'.format(''.join(lines))
        fasp = open('{}/article_spacy_pos/{}'.format(working_dir,filename))
        lines = ['<s> {} </s> '.format(' '.join(k.strip().split())) for k in fasp]
        article_pos = '<d> {}</d>'.format(''.join(lines))
        fasn = open('{}/article_spacy_ner/{}'.format(working_dir,filename))
        lines = ['<s> {} </s> '.format(' '.join(k.strip().split())) for k in fasn]
        article_ner = '<d> {}</d>'.format(''.join(lines))
        ftsl = open('{}/title_spacy_line/{}'.format(working_dir,filename))
        lines = ['<s> {} </s> '.format(' '.join(k.strip().split())) for k in ftsl]
        title_line = '<d> {}</d>'.format(''.join(lines))
        ftsp = open('{}/title_spacy_pos/{}'.format(working_dir,filename))
        lines = ['<s> {} </s> '.format(' '.join(k.strip().split())) for k in ftsp]
        title_pos = '<d> {}</d>'.format(''.join(lines))
        ftsn = open('{}/title_spacy_ner/{}'.format(working_dir, filename))
        lines = ['<s> {} </s> '.format(' '.join(k.strip().split())) for k in ftsn]
        title_ner = '<d> {}</d>'.format(''.join(lines))
        fhsl = open('{}/highlight_spacy_line/{}'.format(working_dir,filename))
        lines = ['<s> {} </s> '.format(' '.join(k.strip().split())) for k in fhsl]
        df['abstract'].append(''.join(lines))
        highlight_line = '<d> {}</d>'.format(''.join(lines))
        fhsp = open('{}/highlight_spacy_pos/{}'.format(working_dir,filename))
        lines = ['<s> {} </s> '.format(' '.join(k.strip().split())) for k in fhsp]
        highlight_pos = '<d> {}</d>'.format(''.join(lines))
        fhsn = open('{}/highlight_spacy_ner/{}'.format(working_dir,filename))
        lines = ['<s> {} </s> '.format(' '.join(k.strip().split()[0:-1])) for k in fhsn]
        highlight_ner = '<d> {}</d>'.format(''.join(lines))

        fw.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(article_line,article_pos,article_ner,title_line,title_pos,title_ner,highlight_line,highlight_pos,highlight_ner))

    fw.close()

dt = pd.DataFrame.from_dict(df,orient='columns')
train, validate, test = np.split(dt.sample(frac=1), [int(.8*len(dt)), int(.1*len(dt))])

data = {}
data['train'] = train
data['dev'] = validate
data['test'] = test

vocab = Counter()
for filetype in ['train','dev','test']:
    writer = open('{}/{}.bin'.format(root_dir,filetype), 'wb')
    for article, abstract in zip(data[filetype]['article'].values,data[filetype]['abstract'].values):
        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
        tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))
        if filetype in ['train','dev']:
            vocab.update(article.split())
            vocab.update(abstract.split())
    writer.close()

fw = open('{}/vocab-50k'.format(root_dir), 'w')
for (k,v) in vocab.most_common(50000):
    fw.write('{} {}\n'.format(k,v))
fw.close()

