# How to run:
# python cnn_dm_data_maker.py ~/data/cnn_dm/cnn/ ~/data/cnn_dm/cnn/[article,title,highlight] [article,title/highlight]
# This code will generate the processed version of the article, title, and highlights of CNN and DM datasets

import os, sys
from glob import glob
import spacy
nlp = spacy.load('en')
import errno
from multiprocessing import Pool, cpu_counts
from unidecode import unidecode

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.7
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def remove_non_ascii(text):
    try:
        return unicode(unidecode(unicode(text, encoding = "utf-8")))
    except:
        return str(unidecode(str(text)))

def run(fl):
    filename = fl.split('/')[-1]
    text = open(fl).read().strip()
    text = '\n'.join(text.split('\n'))
    doc = nlp.make_doc(remove_non_ascii(text))

    for proc in nlp.pipeline:
        doc = proc(doc)

    fwl = open('{}/{}'.format(linedir, filename),'w')
    fwp = open('{}/{}'.format(posdir, filename),'w')
    fwn = open('{}/{}'.format(nerdir, filename),'w')
    nerlist = ['PERSON','NORP','FACILITY','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LANGUAGE']
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
            #if word.pos_ == "NUM":
            #    word_text= '#'
            #    word_pos = 'NUM'
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
        fwl.write('{} .\n'.format(' '.join(line)))
        fwp.write('{} EOS\n'.format(' '.join(pos)))
        fwn.write('{} EOS\n'.format(' '.join(ner)))
    fwl.close()
    fwp.close()
    fwn.close()

basedir = sys.argv[1]
article_dir = sys.argv[2]
article_title = sys.argv[3]
linedir = '{}/{}_spacy_line/'.format(basedir, article_title)
posdir = '{}/{}_spacy_pos/'.format(basedir, article_title)
nerdir = '{}/{}_spacy_ner/'.format(basedir, article_title)

mkdir_p(linedir)
mkdir_p(posdir)
mkdir_p(nerdir)

filelist = glob('{}/*'.format(article_dir))
print('processing {} files...'.format(len(filelist)))

pool = Pool(cpu_counts())
pool.map(run, filelist)
pool.close()
