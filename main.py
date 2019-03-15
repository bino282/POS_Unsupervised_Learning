from HMM import HMM
from utils import *

p = load_probabilities("prob_vector.pickle")
h = HMM(p)
sequences = load_corpus('./data.txt')
h.update(sequences,cutoff_value=0.001)