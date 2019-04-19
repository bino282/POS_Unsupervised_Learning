from utils import *
import pickle
corpus = load_corpus('./data/vlsp/train')
vocab_list,vocab2idx,idx2voc = build_vocab(corpus)
fw = open("./data/vocab.txt",'w',encoding='utf-8')
for w in vocab_list:
    fw.write(w)
    fw.write('\n')
fw.close()
observation_list = []
for doc in corpus:
    doc_cv = convert_text(doc,vocab_list)
    if(len(doc_cv) > 0):
        observation_list.append(doc_cv)
with open('./data/obs.pickle','wb') as f:
    pickle.dump(observation_list,f)