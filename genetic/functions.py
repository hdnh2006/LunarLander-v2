#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:42:36 2024

@author: henry
"""


import copy
import torch.nn as nn
import numpy as np
import logging
from utils.functions import return_average_score

class LunarLanderAI(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                        nn.Linear(8,128, bias=True), # Inputs provided by environment
                        nn.ReLU(),
                        nn.Linear(128,4, bias=True), # 4 possible movements in output
                        nn.Softmax(dim=1) # Logistic function generalization, it gives the probability of each movement
                        )

                
        def forward(self, inputs):
            x = self.fc(inputs)
            return x


class GeneticAlgorithm:
    def __init__(self, num_agents = 500, runs_gen = 1, runs_elite = 3, mutation_power = 0.02, elite_index = None):
        self.num_agents = num_agents
        self.runs_gen = runs_gen
        self.runs_elite = runs_elite
        self.mutation_power = mutation_power
        self.elite_index = elite_index
        self.only_consider_top_n = 10
        self.agents = self.return_random_agents()
        
    
    def return_random_agents(self):
        
        agents = []
        for _ in range(self.num_agents):
            
            agent = LunarLanderAI()
            
            for param in agent.parameters():
                param.requires_grad = False
                
            agent = self.init_weights(agent)
            agents.append(agent)
            
        return agents
        
    @staticmethod  
    def init_weights(agent):
        
            # nn.Conv2d weights are of shape [16, 1, 3, 3] i.e. # number of filters, 1, stride, stride
            # nn.Conv2d bias is of shape [16] i.e. # number of filters
            
            # nn.Linear weights are of shape [32, 24336] i.e. # number of input features, number of output features
            # nn.Linear bias is of shape [32] i.e. # number of output features
            
            if ((type(agent) == nn.Linear) | (type(agent) == nn.Conv2d)):
                nn.init.xavier_uniform(agent.weight)
                agent.bias.data.fill_(0.00)
            
            return agent
    
    
    @staticmethod
    def crossover(mother_agent,father_agent):
        
        child_agent = copy.deepcopy(mother_agent)
        
        dim_father=[]
        for j in range(len(list(father_agent.parameters()))):
            dim_father.append(list(father_agent.parameters())[j].shape)
    
    
        
        for param in child_agent.parameters():
            for j in range(len(dim_father)):
                if(dim_father[j]==param.shape):
                    idx=j
            
            
            if(len(param.shape)==4): #weights of Conv2D
    
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        for i2 in range(param.shape[2]):
                            for i3 in range(param.shape[3]):
                                if np.random.uniform(0,1) <= 0.5:
                                    param[i0][i1][i2][i3]= list(father_agent.parameters())[idx][i0][i1][i2][i3]
                                    
                                        
    
            elif(len(param.shape)==2): #weights of linear layer
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        if np.random.uniform(0,1) <= 0.5:
                            param[i0][i1]= list(father_agent.parameters())[idx][i0][i1]
    
                            
    
            elif(len(param.shape)==1): #biases of linear layer or conv layer
                for i0 in range(param.shape[0]):
                    if np.random.uniform(0,1) <= 0.5:
                        param[i0]= list(father_agent.parameters())[idx][i0]
        
        return child_agent
    

    
    def mutate(self, agent):

        child_agent = copy.deepcopy(agent)
        
        mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
        
        for param in child_agent.parameters():
            
            if(len(param.shape)==4): #weights of Conv2D

                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        for i2 in range(param.shape[2]):
                            for i3 in range(param.shape[3]):
                                
                                param[i0][i1][i2][i3]+= mutation_power * np.random.randn()
                                    
                                        

            elif(len(param.shape)==2): #weights of linear layer
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        
                        param[i0][i1]+= mutation_power * np.random.randn()
                            

            elif(len(param.shape)==1): #biases of linear layer or conv layer
                for i0 in range(param.shape[0]):
                    
                    param[i0]+=mutation_power * np.random.randn()
            


        return child_agent
    
    
    def add_elite(self, sorted_parent_indexes, elite_index=None, show_game = False):
        
        candidate_elite_index = sorted_parent_indexes[:self.only_consider_top_n]
        
        if(elite_index is not None):
            candidate_elite_index = np.append(candidate_elite_index,[elite_index])
            
        top_score = None
        top_elite_index = None
        
        logging.info("Start: Playing candidates to elite...")
        for i in candidate_elite_index:
            
            # Just show some of them
            # if(i%10==0):
            #     show=True
            # else:
            #     show=False
            
            score = return_average_score(self.agents[i], self.runs_elite, show_game)
            #print("Score for elite i ", i, " is on average", score)
            
            if(top_score is None):
                top_score = score
                top_elite_index = i
            elif(score > top_score):
                top_score = score
                top_elite_index = i
        
        logging.info("End: Playing candidates to elite...")    
        logging.info(f'Elite selected with index {top_elite_index} and average score {top_score}')
        
        child_agent = copy.deepcopy(self.agents[top_elite_index])
        return child_agent

   
    
    

    def return_children(self, sorted_parent_indexes, elite_index, show_game):
        
        children_agents = []
        
        logging.info("Start: Crossing and Muting agents...")
        
        max_idx=1
        while max_idx<=len(sorted_parent_indexes):
            
            for i in range(max_idx):
                children = self.crossover(mother_agent=self.agents[sorted_parent_indexes[i]], father_agent=self.agents[max_idx])
                children_agents.append(self.mutate(children))
                if(len(children_agents)>=(self.num_agents-1)):
                    break
            
            max_idx=max_idx+1
               
        
        logging.info("End: Crossing and Muting agents...")
        
        
        mutated_agents=[]
        
        #now add one elite
        elite_child = self.add_elite(sorted_parent_indexes, elite_index, show_game)
        children_agents.append(elite_child)
        elite_index = len(children_agents) -1 # This is the last one
        
        return children_agents, elite_index









