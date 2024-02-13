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
import logging

from genetic.functions import GeneticAlgorithm
from utils.functions import run_agents_n_times, save_generation_data

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(message)s',
)


# Argument parser setup
parser = argparse.ArgumentParser(description='Run LunarLander AI agent with Deep Neuroevolution Strategies.')
parser.add_argument('--num-agents', type=int, default=500, help='Number of agents to initialize')
parser.add_argument('--generations', type=int, default=500, help='Number of generations to evolve')
parser.add_argument('--runs-gen', type=int, default=1, help='Number of runs for the game in each generation (default: 1)')
parser.add_argument('--runs-elite', type=int, default=1, help='Number of runs for the game while eliting (default: 3)')
parser.add_argument('--top-limit', type=int, help='Number of top agents to consider as parents')
parser.add_argument('--show-game', type=bool, default=False, help='Whether to render the game environment')
args = parser.parse_args()

# variables
num_agents, generations, runs_gen, runs_elite, top_limit, show_game = args.num_agents, args.generations, args.runs_gen, args.runs_elite, args.top_limit, args.show_game

# Apply warnings filter
warnings.filterwarnings('ignore')

# disable gradients as we will not use them
torch.set_grad_enabled(False)

# This comment refers to solving the equation n(n+1)/2 = num_agents,
# which determines the required number of iterations for a crossover process in a genetic algorithm.
top_limit = int((-1 + np.sqrt(1 + 4*2*num_agents))/2)+1 

# run evolution until X generations
elite_index = None

# Initialize data frames and genetic algorithm
df_rewards_performance=pd.DataFrame(columns=["generation","reward_best_agent","average_top_elite","average_all_agents",
                       "median_top_elite","median_all_agents", "sd_top_elite","sd_all_agents"])
    

df_rewards_cummulative_performance=pd.DataFrame(columns=["generation","cumm_reward_best_agent","cumm_average_top_elite",
                                                         "cumm_average_all_agents","cumm_median_top_elite","cumm_median_all_agents"])


ga_instance = GeneticAlgorithm(num_agents = 500, runs_gen = 1, runs_elite = 3, mutation_power = 0.02, elite_index = None)

for generation in range(0,generations):

    # return rewards of agents
    rewards = run_agents_n_times(ga_instance.agents, runs_gen, show_game=False) # return average of 'runs_gen' runs
    
    # Save values each 20 generations 
    save_generation_data(generation, ga_instance, rewards, df_rewards_performance, df_rewards_cummulative_performance)
    
    # sort by rewards
    # reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
    sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] 
    top_rewards = []

    for best_parent in sorted_parent_indexes:
        top_rewards.append(rewards[best_parent])
    
    #----------------- Start: performance metrics -----------------------#
    
    df_rewards_performance.loc[generation] = [generation,top_rewards[0], np.mean(top_rewards),np.mean(rewards),
                               np.median(top_rewards),np.median(rewards),np.std(top_rewards),np.std(rewards)]
    
        
    cumulative_values = df_rewards_cummulative_performance.iloc[-1, 1:6].values if generation != 0 else 0
    df_rewards_cummulative_performance.loc[generation] = [generation] + list(df_rewards_performance.iloc[generation, 1:6].values + cumulative_values)

    plt.figure()
    plt.plot(df_rewards_performance.generation,df_rewards_performance.average_top_elite,"go-")
    #norm_average=(df_rewards_cummulative_performance.cumm_average_top_elite-np.mean(df_rewards_cummulative_performance.cumm_average_top_elite))/np.std(df_rewards_cummulative_performance.cumm_average_top_elite)
    #plt.plot(df_rewards_performance.generation,norm_average,"go-")
    plt.show()
    
    #----------------- End: performance metrics -----------------------#
    
    logging.info(f'Generation {generation} | Mean rewards all players: {np.mean(rewards)} | Mean of top 5: {np.mean(top_rewards[:5])}')
    logging.info(f'Top {top_limit} scores: {sorted_parent_indexes}')
    logging.info(f'Rewards for top: {top_rewards}')
    
    # setup an empty list for containing children agents
    children_agents, elite_index = ga_instance.return_children(sorted_parent_indexes, elite_index)

    # replace agents by their children
    agents = children_agents
    
