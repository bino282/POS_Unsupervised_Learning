import pickle
import math
def load_corpus(path):
    ret = []
    with open(path) as f:
        for line in f:
            for c in line:
                if c.isalpha():
                    ret.append(c.lower())

                if c.isspace() and ret and not ret[-1].isspace():
                    ret.append(" ")

    ret = "".join(ret)
    return ret.strip()
def load_probabilities(path):
    ret = None
    with open(path,'rb') as f:
        ret = pickle.load(f)
    return ret