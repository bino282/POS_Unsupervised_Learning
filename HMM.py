import json
import os
import sys
import math
import numpy as np

class MyHmm(): # base class for different HMM models
    def __init__(self,vocab,states,pi):
        self.W_a = np.random.rand(2*len(states))
        self.W_b = np.random.rand(2*len(vocab))
        self.states = states
        self.symbols = vocab 
        self.N = len(self.states) 
        self.M = len(self.symbols) 
        self.pi = pi
        self.A = self.cacul_A()
        self.B = self.cacul_B()
        print(self.A)
    
    def get_feature_b(self,w,state):
        init_feat = [ 0 for i in range(2*self.M)]
        if(state == 'B'):
            init_feat[self.symbols.index(w)] = 1
        else:
            init_feat[self.M + self.symbols.index(w)] = 1
        return np.asarray(init_feat)

    def get_feature_a(self,state1,state2):
        if(state1=='B' and state2=='B'):
            init_feat= [1,0,1,0]
        if(state1=='B' and state2=='I'):
            init_feat= [1,0,0,1]
        if(state1=='I' and state2=='B'):
            init_feat= [0,1,1,0]
        if(state1=='I' and state2=='I'):
            init_feat= [0,1,0,1]
        return np.asarray(init_feat)

    def get_b_pro(self,w,state):
        theta_b = math.exp(np.dot(self.W_b,self.get_feature_b(w,state)))    
        return theta_b
    
    def get_a_pro(self,state1,state2):
        theta_a = math.exp(np.dot(self.W_a,self.get_feature_a(state1,state2)))    
        return theta_a

    def cacul_A(self):
        a = {}
        for i in ("B","I"):
            tmp = [self.get_a_pro(i,j) for j in ("B","I")]
            tmp = [tmp[k]/sum(tmp) for k in range(len(tmp))]
            a[i] = {"B":tmp[0],"I":tmp[1]}
        return a


    def cacul_B(self):
        b_1 = {}
        b_2 = {}
        tmp1 = [self.get_b_pro(self.symbols[i],state='B') for i in range(len(self.symbols))]
        tmp1 = [tmp1[i]/sum(tmp1) for i in range(len(tmp1))]
        tmp2 = [self.get_b_pro(self.symbols[i],state='I') for i in range(len(self.symbols))]
        tmp2 = [tmp2[i]/sum(tmp2) for i in range(len(tmp2))]
        for i in range(len(self.symbols)):
            b_1[self.symbols[i]] = tmp1[i]
            b_2[self.symbols[i]] = tmp2[i]
        b = {"B":b_1,"I":b_2}
        return b


    def backward(self, obs):
        self.bwk = [{} for t in range(len(obs))]
        T = len(obs)
        # Initialize base cases (t == T)
        for y in self.states:
            self.bwk[T-1][y] = 1 #self.A[y]["Final"] #self.pi[y] * self.B[y][obs[0]]
        for t in reversed(range(T-1)):
            for y in self.states:
                self.bwk[t][y] = sum((self.bwk[t+1][y1] * self.A[y][y1] * self.B[y1][obs[t+1]]) for y1 in self.states)
        prob = sum((self.pi[y]* self.B[y][obs[0]] * self.bwk[0][y]) for y in self.states)
        return prob,self.bwk

    def forward(self, obs):
        self.fwd = [{}]     
        # Initialize base cases (t == 0)
        for y in self.states:
            self.fwd[0][y] = self.pi[y] * self.B[y][obs[0]]
        # Run Forward algorithm for t > 0
        for t in range(1, len(obs)):
            self.fwd.append({})     
            for y in self.states:
                self.fwd[t][y] = sum((self.fwd[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)
        prob = sum((self.fwd[len(obs) - 1][s]) for s in self.states)
        return prob,self.fwd

    def viterbi(self, obs):
        vit = [{}]
        path = {}     
        # Initialize base cases (t == 0)
        for y in self.states:
            vit[0][y] = self.pi[y] * self.B[y][obs[0]]
            path[y] = [y]
     
        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}     
            for y in self.states:
                (prob, state) = max((vit[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]     
            # Don't need to remember the old paths
            path = newpath
        n = 0           # if only one element is observed max is sought in the initialization values
        if len(obs)!=1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])

    def forward_backward(self, obs): # returns model given the initial model and observations        
        gamma = [{} for t in range(len(obs))] # this is needed to keep track of finding a state i at a time t for all i and all t
        zi = [{} for t in range(len(obs) - 1)]  # this is needed to keep track of finding a state i at a time t and j at a time (t+1) for all i and all j and all t
        # get alpha and beta tables computes
        p_obs = self.forward(obs)
        self.backward(obs)
        # compute gamma values
        for t in range(len(obs)):
            for y in self.states:
                gamma[t][y] = (self.fwd[t][y] * self.bwk[t][y]) / p_obs
                if t == 0:
                    self.pi[y] = gamma[t][y]
                #compute zi values up to T - 1
                if t == len(obs) - 1:
                    continue
                zi[t][y] = {}
                for y1 in self.states:
                    zi[t][y][y1] = self.fwd[t][y] * self.A[y][y1] * self.B[y1][obs[t + 1]] * self.bwk[t + 1][y1] / p_obs
        # now that we have gamma and zi let us re-estimate
        for y in self.states:
            for y1 in self.states:
                # we will now compute new a_ij
                val = sum([zi[t][y][y1] for t in range(len(obs) - 1)]) #
                val /= sum([gamma[t][y] for t in range(len(obs) - 1)])
                self.A[y][y1] = val
        # re estimate gamma
        for y in self.states:
            for k in self.symbols: # for all symbols vk
                val = 0.0
                for t in range(len(obs)):
                    if obs[t] == k :
                        val += gamma[t][y]                 
                val /= sum([gamma[t][y] for t in range(len(obs))])
                self.B[y][k] = val
        return

    def baum_welch(self,observations):
        gammas = []
        gamma_obs = []
        zis = []
        for obs in observations:
            prob_fwd,fwd = self.forward(obs)
            prob_bwk,bwk = self.backward(obs)
            P_O = prob_fwd
            if P_O==0 or P_O=='inf' :
                continue
            # compute gamma(P_q_O) values
            gamma = [{} for t in range(len(obs))]
            gamma_obs.append(obs)
            zi = [{} for t in range(len(obs) - 1)]
            for t in range(len(obs)):
                for y in self.states:
                    gamma[t][y] = (fwd[t][y] * bwk[t][y]) / P_O
                    if t == len(obs) - 1:
                        continue
                    zi[t][y] = {}
                    for y1 in self.states:
                        zi[t][y][y1] = self.fwd[t][y] * self.A[y][y1] * self.B[y1][obs[t + 1]] * self.bwk[t + 1][y1] / P_O
            gammas.append(gamma)
            zis.append(zi)
        C_q1i = {}
        C_qi_o = {}
        C_qi_qj = {}
        for y in self.states:
            C_q1i[y] = sum([gammas[o][0][y] for o in range(len(gammas))])

        
        for y in self.states:
            C_qi_o[y] = {}
            for k in self.symbols:
                C_qi_o[y][k] = 0
        for y in self.states:
            for i in range(len(gammas)):
                for j in range (len(gamma_obs[i])):
                    C_qi_o[y][gamma_obs[i][j]] += gammas[i][j][y]
        for y1 in self.states:
            C_qi_qj[y1]= {}
            for y2 in self.states:
                C_qi_qj[y1][y2] = sum([zis[o][l][y1][y2] for o in range(len(zis)) for l in range(len(zis[o]))])
        C_q1 = sum([C_q1i[y] for y in self.states])
        C_qi = {}
        for y in self.states:
            C_qi[y] = sum([C_qi_qj[y][t] for t in self.states])
        for y in self.states:
            self.pi[y] = C_q1i[y]/C_q1
        # for y1 in self.states:
        #     for y2 in self.states:
        #         self.A[y1][y2] = C_qi_qj[y1][y2]/C_qi[y1]
        # for y in self.states:
        #     for k in self.symbols:
        #         self.B[y][k] = C_qi_o[y][k]/C_qi[y]
        # print("pi : ")
        # print(self.pi)
        # print("transison : ")
        # print(self.A)

        L_W_E = 0
        for d in self.symbols:
            for c in self.states:
                # transison
                L_W_E += C_qi_qj[c][d]* math.log(self.A[c][d])
                # emission
                L_W_E += C_qi_o
       
        return 1 

    def cacul_denta_dct(self,d,c,t):
        denta_d_c_t = 0
        if (t=='b'):
            sum_d = 0
            for w in self.symbols:
                sum_d += np.dot(self.B[c][w],self.get_feature_b(w,c))
            denta_d_c_t = self.get_feature_b(d,c) - sum_d
        if(t=='a'):
            sum_d = 0
            for c1 in self.states:
                sum_d += np.dot(self.A[c1][c],self.get_feature_b(c1,c))
            denta_d_c_t = self.get_feature_a(c,d) - sum_d
        return denta_d_c_t




         

        
        