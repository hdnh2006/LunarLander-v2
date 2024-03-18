#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:26:57 2024

@author: henry
"""

import argparse
import warnings
import numpy as np
import torch
import pickle
import gymnasium as gym
import datetime

# Apply warnings filter
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Play your LunarLander AI agent in gym environment')
parser.add_argument('--agents', help='agents.pkl file')
parser.add_argument('--rewards', help='rewards.pkl file')
parser.add_argument('--runs', type=int, default=10, help='Number of runs the agent will be ran')
parser.add_argument('--show-game', action="store_true", help='Whether to render the game environment')
args = parser.parse_args()

runs, show_game = args.runs, args.show_game

# Read agents
with open(args.agents, 'rb') as config_dictionary_file:
 
    # Load nn
    agents = pickle.load(config_dictionary_file)
 
    # After config_dictionary is read from file
    print(agents)
    
# Read rewards
with open(args.rewards, 'rb') as config_dictionary_file:
 
    # Load rewards
    rewards = pickle.load(config_dictionary_file)
 
    # After config_dictionary is read from file
    print(rewards)


def run_agent(agents):
    
    for i in range(runs):

        agent=agents[np.argmax(rewards)]
        
        env = gym.make("LunarLander-v2", render_mode = "human" if show_game else None)

        agent.eval()
    
        observation = env.reset()[0]
        r=0
        s=0
        
        tini=datetime.datetime.now()

        while True:

            inp = torch.tensor(observation).type('torch.FloatTensor').view(1,-1)
            output_probabilities = agent(inp).detach().numpy()[0]
            action = np.random.choice([0,1,2,3], 1, p=output_probabilities).item()

            observation, reward, done, info, _ = env.step(action)
                
            r=r+reward
        
            s=s+1

            tfin=datetime.datetime.now()
            
            # if reward is too low or the spaceship keeps flying => set to done.
            if((r<-250) or ((tfin-tini).seconds >120) ):
                done=True

            if(done):
                env.close()
                break
        
        

                
run_agent(agents)
