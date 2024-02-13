#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:28:50 2024

@author: henry
"""



import torch
import datetime
import pickle
import numpy as np
import gymnasium as gym

def run_agents_n_times(agents, runs):
    """
    Runs multiple agents a specified number of times and calculates their average score.
    
    Parameters:
        agents (list): List of agent models to be evaluated.
        runs (int): Number of runs to perform for each agent.
        show_game (bool): Flag to render the game environment or not.
        
    Returns:
        list: Average scores of the agents over the specified number of runs.
    """
    avg_score = [return_average_score(agent, runs) for agent in agents]
    return avg_score


def return_average_score(agent, runs, show_game=False):
    """
    Calculates the average score of a single agent over a specified number of runs.
    
    Parameters:
        agent: The agent model to be evaluated.
        runs (int): Number of runs to perform.
        show_game (bool): Flag to render the game environment or not.
        
    Returns:
        float: Average score of the agent over the specified number of runs.
    """
    score = sum(run_agents([agent], show_game)[0] for _ in range(runs)) / runs
    return score


def run_agents(agents,show_game=True):
    """
    Executes a series of game runs for the given agents in the LunarLander-v2 environment.
    
    Parameters:
        agents (list): List of agent models to run in the environment.
        show_game (bool): If true, renders the game environment.
        
    Returns:
        list: Rewards accumulated by each agent.
    """
    
    reward_agents = []
    
    env = gym.make("LunarLander-v2", render_mode = "human" if show_game else None)

    for agent in agents:
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
            
        reward_agents.append(r)        
    
    return reward_agents



def save_generation_data(generation, ga_instance, rewards, df_rewards_performance, df_rewards_cummulative_performance):
    """
    Saves the agents, rewards, and performance dataframes at specific generation intervals.
    
    Parameters:
        generation (int): The current generation number.
        ga_instance: The GeneticAlgorithm instance containing the current generation's agents.
        rewards (list): The list of rewards for the current generation.
        df_rewards_performance (DataFrame): DataFrame containing performance metrics.
        df_rewards_cummulative_performance (DataFrame): DataFrame containing cumulative performance metrics.
    """
    if generation % 20 == 0 and generation != 0:
        with open(f'agents_generation_{generation}.pkl', 'wb') as agents_file:
            pickle.dump(ga_instance.agents, agents_file)
        
        with open(f'rewards_generation_{generation}.pkl', 'wb') as rewards_file:
            pickle.dump(rewards, rewards_file)
             
        with open(f'df_rewards_performance_generation_{generation}.pkl', 'wb') as performance_file:
            pickle.dump(df_rewards_performance, performance_file)
        
        with open(f'df_rewards_cummulative_performance_generation_{generation}.pkl', 'wb') as cumulative_performance_file:
            pickle.dump(df_rewards_cummulative_performance, cumulative_performance_file)
        
        print(f'Results succesfully saved for generation{generation}')
        
        return True


