# This will be the main python
import numpy as np
import random

def step(N):

    pass

def generate_adj_matrix(N, y, epsilon, probs, m, agents, beta):
    adjacency_matrix = np.zeros(N,N)

    for i in range(N):
        a = probs[i]
        actual_prob = (1 - y) / (1 - epsilon**(1-y)) * a**(-y)

        # Calculate denominator of equation 3
        total_sum = 0
        for j in range(N):
            if j == i:
                continue
            total_sum += abs(agents[i] - agents[j])**-beta
        
        r = np.random.rand()
        # If the agent i is active
        if (actual_prob > r):
            agents_influened = 0
            while agents_influened <= m:
                agent_index = np.randint(0, N)
                
                influence_prob = abs(agents[i] - agents[agent_index])**-beta / total_sum

                # If agent i succesfully influences agent 'agent_index'
                if influence_prob > np.random.rand():
                    adjacency_matrix[agent_index, i] = 1
                    agents_influened += 1

def initialize(n, epsilon):
    agents = np.zeros(n)

    for i in range(n):
        agents[i] = np.randint(-1,1)

    agent_probabilities = np.zeros(n)
    for i in range(n):
        agent_probabilities[i] = np.randint(epsilon,1)
    
    