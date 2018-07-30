# How to run: python newsroom_data_maker.py $HOME/data/newsroom/
# The directory must contain the following files: [train,dev,test].data

# This code will tokenize and detect Named-Entities and concat them with "_".
# Therefore, given the following sentence:
# --President Trump said he based his decision on "open hostility" from North Korea.--
# it will return:
# --president_trump said he based his decision on " open hostility " from north_korea EOS--

from tqdm import tqdm
import os, sys
import struct
import spacy
nlp = spacy.load('en')
from unidecode import unidecode
import random
from multiprocessing import Pool, cpu_count
from newsroom import jsonl
from newsroom.analyze import Fragments
from spacy.tokenizer import Tokenizer
tokenizer = Tokenizer(nlp.vocab)
from nltk.tokenize import word_tokenize
import struct
from tensorflow.core.example import example_pb2
from collections import Counter

def remove_non_ascii(text):
    try:
        return unicode(unidecode(unicode(text, encoding = "utf-8")))
    except:
        return str(unidecode(str(text)))

def pre_processing(text):
    doc = nlp.make_doc(remove_non_ascii(text))
    for name, proc in nlp.pipeline:
        doc = proc(doc)
    nerlist = ['PERSON','NORP','FACILITY','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LANGUAGE']
    line_text = []
    pos_text = []
    ner_text = []
    for i,s in enumerate(doc.sents):
        line = []
        pos = []
        ner = []
        phrase = []
        prev_ent_type = ''
        for word in s:
            word_text = word.text.lower()
            word_pos = word.pos_
            if word.pos_ == "PUNCT":
                if phrase != []:
                    line.append('_'.join(phrase))
                    pos.append('NER')
                    ner.append(prev_ent_type)
                    phrase = []
                    prev_ent_type = word.ent_type_
                continue
            if word.pos_ == "SPACE":
                if phrase != []:
                    line.append('_'.join(phrase))
                    pos.append('NER')
                    ner.append(prev_ent_type)
                    phrase = []
                    prev_ent_type = word.ent_type_
                continue
            '''
            if word.pos_ == "NUM":
                word_text= '#'
                word_pos = 'NUM'
            '''
            if word.ent_type_ != "" and word.ent_type_ in nerlist:
                if prev_ent_type == word.ent_type_:
                    phrase.append(word_text)
                elif prev_ent_type!='':
                    line.append('_'.join(phrase))
                    pos.append('NER')
                    ner.append(prev_ent_type)
                    phrase = [word_text]
                    prev_ent_type = word.ent_type_
                else:
                    prev_ent_type = word.ent_type_
                    phrase.append(word_text)
                continue
            if phrase != []:
                line.append('_'.join(phrase))
                pos.append('NER')
                ner.append(prev_ent_type)
                phrase = []
            line.append(word_text)
            pos.append(word_pos)
            ner.append(word.ent_type_)
            prev_ent_type = word.ent_type_
        line_text.append('{} EOS'.format(' '.join(line)))
        pos_text.append('{} EOS'.format(' '.join(pos)))
        ner_text.append('{} EOS'.format(' '.join(ner)))
    return '\n'.join(line_text), '\n'.join(pos_text), '\n'.join(ner_text)

def run(entry):
    text = entry['text']
    summary = entry['summary']
    text = ' '.join([_.text for _ in tokenizer(remove_non_ascii(text))])
    summary = ' '.join([_.text for _ in tokenizer(remove_non_ascii(summary))])
    text = nlp(text)
    summary = nlp(summary)
    text = '\n'.join([' '.join([_.text for _ in s]) for s in text.sents])
    summary = '\n'.join([' '.join([_.text for _ in s]) for s in summary.sents])
    # run pre-processing
    line_text, pos_text, ner_text = pre_processing(text)
    line_summary, pos_summary, ner_summary = pre_processing(summary)
    entry['processed'] = {}
    entry['processed']['text'] = line_text
    entry['processed']['pos_text'] = pos_text
    entry['processed']['ner_text'] = ner_text
    entry['processed']['summary'] = line_summary
    entry['processed']['pos_summary'] = pos_summary
    entry['processed']['ner_summary'] = ner_summary
    entry['text'] = text.lower()
    entry['summary'] = summary.lower()
    return entry

root_dir = sys.argv[1]

filetypes = ['train', 'dev', 'test']

vocab = Counter()
pvocab = Counter()

for filetype in filetypes:
    filename = "{}/{}.data".format(root_dir, filetype)
    with jsonl.open(filename, gzip = True) as _file:
        entries = [_ for _ in _file]

    print("processing {} files...".format(len(entries)))

    pbar = tqdm(total=len(entries))

    processed_entries = []
    pool = Pool(cpu_count())
    for _ in pool.imap_unordered(run, entries):
        processed_entries.append(_)
        pbar.update(1)

    pool.close()

    assert len(processed_entries) == len(entries)

    # creating input for RLSeq2Seq model using tensorflow example
    pwriter = open('{}/processed_{}.bin'.format(root_dir,filetype), 'wb')
    rwriter = open('{}/{}.bin'.format(root_dir, filetype), 'wb')
    with jsonl.open('{}/processed_{}.data'.format(root_dir, filetype), gzip = True) as processed_dataset_file:
        for entry in processed_entries:
            # pointer-generator consistent input
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([entry['processed']['text'].encode()])
            abstract = ' '.join(['<s> {} </s>'.format(_) for _ in entry['processed']['summary'].split('\n')]).encode()
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            pwriter.write(struct.pack('q', str_len))
            pwriter.write(struct.pack('%ds' % str_len, tf_example_str))

            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([entry['text'].encode()])
            abstract = ' '.join(['<s> {} </s>'.format(_) for _ in entry['summary'].split('\n')]).encode()
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            rwriter.write(struct.pack('q', str_len))
            rwriter.write(struct.pack('%ds' % str_len, tf_example_str))
            if filetype in ['train', 'dev']:
                vocab.update(entry['text'].split())
                vocab.update(entry['summary'].split())
                pvocab.update(entry['processed']['text'].split())
                pvocab.update(entry['processed']['text'].split())

            processed_dataset_file.append(entry)

    pwriter.close()
    rwriter.close()

fw = open('{}/vocab-50k'.format(root_dir), 'w')
for (k,v) in vocab.most_common(50000):
    fw.write('{} {}\n'.format(k,v))
fw.close()

fw = open('{}/processed_vocab-50k'.format(root_dir), 'w')
for (k,v) in pvocab.most_common(50000):
    fw.write('{} {}\n'.format(k,v))
fw.close()
