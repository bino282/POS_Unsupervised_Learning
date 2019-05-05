import json
import os
import sys
import numpy as np
from HMM import MyHmm
import pickle
from utils import *



N = 2
pi ={"B":2.0/3,"I":1.0/3}

a = {}
for i in ("B","I"):
    tmp = {"B":1.0/2,"I":1.0/2}
    a[i] = tmp

vocab_list = []
with open("./data/vocab.txt",'r',encoding='utf-8') as lines:
    for line in lines:
        vocab_list.append(line.strip())

b_1 = {}
b_2 = {}
tmp1 = [1.0/len(vocab_list) for i in range(len(vocab_list))]
tmp2 = [1.0/len(vocab_list) for i in range(len(vocab_list))]
for i in range(len(vocab_list)):
    b_1[vocab_list[i]] = tmp1[i]
    b_2[vocab_list[i]] = tmp2[i]
b = {"B":b_1,"I":b_2}

with open('./data/obs.pickle','rb') as f:
    observation_list= pickle.load(f)
print(pi)
if __name__ == '__main__':
    # test the forward algorithm and backward algorithm for same observations and verify they produce same output
    # we are computing P(O|model) using these 2 algorithms.
    # hmm = MyHmm([a,b,pi])
    
    # total1 = total2 = 0 # to keep track of total probability of distribution which should sum to 1
    # for obs in observation_list:
    #     if(len(obs)==0):
    #         continue
    #     p1 = hmm.forward(obs)
    #     p2 = hmm.backward(obs)
    #     total1 += p1
    #     total2 += p2
    #     print ("Observations = ", obs, " Fwd Prob = ", p1, " Bwd Prob = ", p2, " total_1 = ", total1, " total_2 = ", total2)

    # #test the Viterbi algorithm
    # observations = observation_list[0] + observation_list[1] + observation_list[2]  # you can set this variable to any arbitrary length of observations
    # prob, hidden_states = hmm.viterbi(observations)
    # print ("Max Probability = ", prob, " Hidden State Sequence = ", hidden_states)

    hmm = MyHmm(vocab_list,["B","I"],pi)

    print ("Learning the model through Forward-Backward Algorithm for the observations")
    for epoch in range(1000):
        print("Running on epoch : {} ..................".format(epoch))
        hmm.baum_welch(observation_list)
        observations = observation_list[0] # you can set this variable to any arbitrary length of observations
        prob, hidden_states = hmm.viterbi(observations)
        print(observations)
        print ("Max Probability = ", prob, " Hidden State Sequence = ", hidden_states)

    print ("The new model parameters after 1 iteration are: ")
    # print ("A = ", hmm.A)
    # print ("B = ", hmm.B)
    # print ("pi = ", hmm.pi)

    # test the Viterbi algorithm
    # observations = observation_list[0] + observation_list[1] + observation_list[2]  # you can set this variable to any arbitrary length of observations
    # prob, hidden_states = hmm.viterbi(observations)
    # print ("Max Probability = ", prob, " Hidden State Sequence = ", hidden_states)