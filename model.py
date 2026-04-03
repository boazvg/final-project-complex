# This will be the main python file
import numpy as np
import random
import math

def step(N, y, epsilon, m, agents, beta, K, a, dt):
    dx_dt = np.zeros(N)
    adj_matrix = generate_adj_matrix(N, y, epsilon, m, agents, beta)
    for i in range(N):
        dx_dt[i] = function_1(i, K, N, adj_matrix, a, agents)
    for i in range(N):
        agents[i] += dt*dx_dt[i]
    return(agents)

def function_1(x_index, K, N,  adj_matrix, a, agents):
    x = agents[x_index]
    total_sum = 0
    for j in range(N):
        total_sum += adj_matrix[x_index, j]*math.tanh(a*agents[j])
    result = -x + K*total_sum
    return(result)

def generate_adj_matrix(N, y, epsilon, m, agents, beta):
    adjacency_matrix = np.zeros((N,N))
    probs = np.zeros(N)
    for i in range(N):
        probs[i] = random.uniform(epsilon, 1)
    for i in range(N):
        a = probs[i]
        actual_prob = (1 - y) / (1 - epsilon**(1-y)) * a**(-y)
        r = np.random.rand()
        # If the agent is active
        if (actual_prob > r):
            # Calculate denominator of equation 3
            total_sum = 0
            for j in range(N):
                if j == i:
                    continue
                total_sum += abs(agents[i] - agents[j])**-beta
            agents_influened = 0
            while agents_influened <= m:
                agent_index = random.randint(0, N-1)
                influence_prob = abs(agents[i] - agents[agent_index])**-beta / total_sum

                # If agent i succesfully influences agent 'agent_index'
                if influence_prob > np.random.rand():
                    adjacency_matrix[agent_index, i] = 1
                    agents_influened += 1
    return(adjacency_matrix)

def initialize(N):
    agents = np.zeros(N)
    for i in range(N):
        agents[i] = random.uniform(-1,1)
    return(agents)

def simulate(N, K, a, epsilon, m, beta, y, time):
    agents = initialize(N)
    agents_copy = agents.copy()
    for i in range(time):
        agents = step(N, y, epsilon, m, agents, beta, K, a, 0.01)
        print(f"{agents[0]} {agents[1]} {agents[2]} {agents[3]} {agents[4]}")
        print("----------------------------------------------------")
    print(f"starting values: {agents_copy[0]} {agents_copy[1]} {agents_copy[2]} {agents_copy[3]} {agents_copy[4]}")
    print(f"ending values: {agents[0]} {agents[1]} {agents[2]} {agents[3]} {agents[4]}")


simulate(100, 3, 3, 0.01, 10, 3, 2.1, 200)