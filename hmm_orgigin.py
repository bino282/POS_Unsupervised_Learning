from nltk.tag import hmm
import nltk
import pickle
import dill
vocab_list = []
with open("./data/vocab.txt",'r',encoding='utf-8') as lines:
    for line in lines:
        vocab_list.append(line.strip())
states = [0,1]
with open('./data/obs.pickle','rb') as f:
    observation_list= pickle.load(f)
unlabeled_sequences = []
for obs in observation_list:
    new_seq = []
    for w in obs:
        new_seq.append((w,""))
    unlabeled_sequences.append(new_seq)
symbols = vocab_list
trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states, symbols=symbols)
tagger = trainer.train_unsupervised(unlabeled_sequences, max_iterations = 10)
with open("hmm.model", 'wb') as f:
    dill.dump(tagger, f)
with open("hmm.model", 'rb') as f:
    hmm_tagger = dill.load(f)
print (hmm_tagger.tag("quầy bán thức ăn đường phố không đảm bảo vệ sinh".split()))

