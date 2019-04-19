import pickle
import math
import os
import re
import json

def load_corpus(path):
    files = os.listdir(path)
    corpus = []
    for file_name in files:
        doc = []
        new_doc = []
        with open(os.path.join(path,file_name),'r',encoding='utf-8') as f:
            for line in f:
                if ('pBody' in line) and ('/' not in line):
                    new_doc = []
                elif ('pBody' in line) and ('/' in line):
                    doc.extend(new_doc)
                else:
                    new_doc.append(line.strip())
        for d in doc:
            d = d.replace("_", " ")
            corpus.append(d)
    return corpus
def load_punctuation():
        punctuation = {'(', ')', ':', '-', ',', '.', '?', '...', '[', ']', '"', ';', '!'}
        return punctuation
def load_stopWords(path):
    stopwords = []
    with open(path,'r',encoding='utf-8') as lines:
        for line in lines:
            stopwords.append(line.strip())
    return stopwords
def load_viSyllables(path):
    viSyllables = []
    with open(path,'r',encoding='utf-8') as lines:
        for line in lines:
            viSyllables.append(line.strip())
    return viSyllables
def check_type_syllable(syllable, syllables_dictionary, punctuation):
    if syllable.lower() in syllables_dictionary:
        return 'VIETNAMESE_SYLLABLE'
    elif syllable in punctuation:
        return 'PUNCT'
    elif syllable.isdigit():
        return 'NUMBER'
    elif syllable.isalpha() is False and syllable.isdigit() is False:
        return 'CODE'
    else:
        return 'FOREIGN_SYLLABLE'
def clear_str(string):
    char_special = '\.|\,|\;|\(|\)|\>|\<|\'|\"|\-|\/|\:|\?|\!|\[|\]|\{|\}'
    str_clean = re.sub('([' + char_special + '])', r' \1 ', string)
    # str_clean = re.sub('[.]', ' ', str_clean)
    str_clean = str_clean.strip()
    str_clean = ' '.join(str_clean.split())
    return str_clean
syllables_vn = load_viSyllables('./data/vi_syllable.txt')
pun = load_punctuation()
def build_vocab(corpus):
    vocab= set()
    # syllables_vn = load_viSyllables('./data/vi_syllable.txt')
    # pun = load_punctuation()
    for doc in corpus:
        for w in doc.split():
            if check_type_syllable(w,syllables_vn,pun) == "VIETNAMESE_SYLLABLE":
                vocab.add(w)
            else:
                continue
    vocab.add("<PUNCT>")
    vocab.add("<CODE>")
    vocab.add("<NUMBER>")
    vocab.add("<FOREIGN_SYLLABLE>")
    vocab_list = list(vocab)
    vocab2idx = {}
    idx2vocab = {}
    for i in range(0,len(vocab_list)):
        vocab2idx[vocab_list[i]] = i
        idx2vocab[i] = vocab_list[i]
    return vocab_list,vocab2idx,idx2vocab

def convert_text(text,vocab):
    text = text.split()
    new_text = []
    for w in text:
        type_syllable = check_type_syllable(w,syllables_vn,pun)
        if(type_syllable=="VIETNAMESE_SYLLABLE"):
            new_text.append(w)
        else:
            new_text.append('<{}>'.format(type_syllable))
    return new_text

corpus = load_corpus('./data/vlsp/train')