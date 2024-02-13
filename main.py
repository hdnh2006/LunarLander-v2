#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:42:36 2024

@author: henry
"""

import argparse
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from genetic.functions import GeneticAlgorithm
from utils.functions import run_agents_n_times


# Argument parser setup
parser = argparse.ArgumentParser(description='Run LunarLander AI agent with Deep Neuroevolution Strategies.')
parser.add_argument('--num-agents', type=int, default=500, help='Number of agents to initialize')
parser.add_argument('--generations', type=int, default=500, help='Number of generations to evolve')
parser.add_argument('--runs-gen', type=int, default=1, help='Number of runs for the game in each generation (default: 1)')
parser.add_argument('--runs-elite', type=int, default=1, help='Number of runs for the game while eliting (default: 3)')
parser.add_argument('--top-limit', type=int, help='Number of top agents to consider as parents')
parser.add_argument('--show-game', type=bool, default=False, help='Whether to render the game environment')
args = parser.parse_args()

num_agents, generations, runs_gen, runs_elite, top_limit, show_game = args.num_agents, args.generations, args.runs_gen, args.runs_elite, args.top_limit, args.show_game

# Apply warnings filter
warnings.filterwarnings('ignore')



#------------------------------------------------------------------------------------------#

# game_actions = 2 #2 actions possible: left or right

# disable gradients as we will not use them
torch.set_grad_enabled(False)

# solucion de la ecuacion de segundo grado n(n+1)/2= num_agents, se toma este numero de esta manera
# por el bucle que se realiza en el cruce.
top_limit = int((-1 + np.sqrt(1 + 4*2*num_agents))/2)+1 


# run evolution until X generations
elite_index = None

df_rewards_performance=pd.DataFrame(columns=["generation","reward_best_agent","average_top_elite","average_all_agents",
                       "median_top_elite","median_all_agents", "sd_top_elite","sd_all_agents"])
    

df_rewards_cummulative_performance=pd.DataFrame(columns=["generation","cumm_reward_best_agent","cumm_average_top_elite",
                                                         "cumm_average_all_agents","cumm_median_top_elite","cumm_median_all_agents"])


GenAlgProc = GeneticAlgorithm(num_agents = 500, runs_gen = 1, runs_elite = 3, mutation_power = 0.02, elite_index = None)

for generation in range(0,generations):

    # return rewards of agents
    rewards = run_agents_n_times(GenAlgProc.agents, runs_gen, show_game=False) # return average of 'runs_gen' runs
    
    #Guardamos cada 20 generaciones las recompensas y los agentes
    if(generation%20==0 and generation!=0):
        with open(f'agents_generation_{str(generation)}.pkl', 'wb') as my_save_file1:
            pickle.dump(GenAlgProc.agents, my_save_file1)
        
        with open(f'rewards_generation_{str(generation)}', 'wb') as my_save_file2:
            pickle.dump(rewards, my_save_file2)
             
        with open('df_rewards_performance_generation_{str(generation)}', 'wb') as my_save_file3:
            pickle.dump(df_rewards_performance, my_save_file3)
        
        with open('df_rewards_cummulative_performance_generation_{str(generation)}', 'wb') as my_save_file4:
            pickle.dump(df_rewards_cummulative_performance, my_save_file4)
    

    # sort by rewards
    # reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
    sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] 
    top_rewards = []

    for best_parent in sorted_parent_indexes:
        top_rewards.append(rewards[best_parent])
    
    
    #----------------- Start: Store performance metrics -----------------------#
    
    
    df_rewards_performance.loc[generation] = [generation,top_rewards[0], np.mean(top_rewards),np.mean(rewards),
                               np.median(top_rewards),np.median(rewards),np.std(top_rewards),np.std(rewards)]
    
        
    cumulative_values = df_rewards_cummulative_performance.iloc[-1, 1:6].values if generation != 0 else 0
    df_rewards_cummulative_performance.loc[generation] = [generation] + list(df_rewards_performance.iloc[generation, 1:6].values + cumulative_values)

    
    
    
    plt.figure()
    #plt.plot(df_rewards_performance.generation,df_rewards_performance.average_top_elite,"go-")
    norm_average=(df_rewards_cummulative_performance.cumm_average_top_elite-np.mean(df_rewards_cummulative_performance.cumm_average_top_elite))/np.std(df_rewards_cummulative_performance.cumm_average_top_elite)
    plt.plot(df_rewards_performance.generation,norm_average,"go-")
    plt.show()
    
    #----------------- End: Store performance metrics -----------------------#
    
    
    print("Generation ", generation, " | Mean rewards all players: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
    #print(rewards)
    print("Top ",top_limit," scores", sorted_parent_indexes)
    print("Rewards for top: ",top_rewards)
    
    # setup an empty list for containing children agents
    children_agents, elite_index = GenAlgProc.return_children(sorted_parent_indexes, elite_index)

    # kill all agents, and replace them with their children
    agents = children_agents
    
