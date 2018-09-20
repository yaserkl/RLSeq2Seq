# The directory that contains all *.story files: ~/data/cnn_dm/cnn/
# How to run to extract (article, title) pairs:
# python cnn_dm_downloader.py ~/data/cnn_dm/cnn/ ~/data/cnn_dm/cnn/processed/ article
# How to run to extract the highlights:
# python cnn_dm_downloader.py ~/data/cnn_dm/cnn/ ~/data/cnn_dm/cnn/processed/ highlight
# It will generate three directories as follows: ~/data/cnn_dm/cnn/processed/[article,title,highlight]


from glob import glob
from unidecode import unidecode
from multiprocessing import Pool, cpu_counts
import os, sys
from newspaper import Article # require python 3 for this

try:
    reload(sys)
    sys.setdefaultencoding('utf-8') 
except:
    pass
from chardet.universaldetector import UniversalDetector

def encoding_detector(filename):
    detector = UniversalDetector()
    for line in open(filename, 'rb'):
        detector.feed(line)
        if detector.done: break
    detector.close()
    return detector.result['encoding']

def remove_non_ascii(text):
    try:
        return unidecode(unicode(text, encoding = "utf-8"))
    except:
        return unidecode(text)

def run(param):
    (article_dir, title_dir, html_path) = param
    try:
        raw_html = open(html_path, encoding="ascii", errors="surrogateescape").read().strip()
    except:
        raw_html = open(html_path, encoding=encoding_detector(html_path), errors="surrogateescape").read().strip()

    id = html_path.split('/')[-1].split('.')[0]
    a = Article('http:/www.dummy.com', language='en')
    a.download(input_html=raw_html)
    a.parse()
    title = a.title
    text = a.text
    title = remove_non_ascii(title)
    text = remove_non_ascii(text)
    fw = open('{}/{}'.format(article_dir, id),'w',encoding='utf-8')
    fw.write(text)
    fw.close()
    fw = open('{}/{}'.format(title_dir, id),'w',encoding='utf-8')
    fw.write(title)
    fw.close()

def extract_highlight(param):
    (indir, outdir, id) = param
    f = open('{}/{}.story'.format(indir, id))
    lines = f.readlines()
    highlights = []
    for i, line in enumerate(lines):
        if "@highlight" in line.strip():
            try:
                highlights.append('{}.'.format(lines[i+2].strip()))
            except:
                continue
    fw = open('{}/{}'.format(outdir, id),'w')
    fw.write('\n'.join(highlights))
    fw.close()

indir = sys.argv[1]
outdir = sys.argv[2]
mode = sys.argv[3] # highlight/article

if mode == 'article':
    article_dir = '{}/articles'.format(outdir)
    title_dir = '{}/title'.format(outdir)
    if not os.path.exists(article_dir):
        os.makedirs(article_dir)
    if not os.path.exists(title_dir):
        os.makedirs(title_dir)

    params = [(article_dir, title_dir, k) for k in glob('{}/*.html'.format(indir))]
    print('processing {} files...'.format(len(params)))

    pool = Pool(cpu_counts())
    pool.map(run, params, 1000)
    pool.close()
else:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    params = [(indir, outdir, k.split('/')[-1].split('.')[0]) for k in glob('{}/*.story'.format(indir))]
    print('processing {} files...'.format(len(params)))
    pool = Pool(cpu_counts())
    pool.map(extract_highlight, params, 1000)
    pool.close()
