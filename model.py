# This will be the main python file
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def step(N, agent_activation_probabilities, agent_opinions, r, m, beta, K, alpha, dt):
    dx_dt = np.zeros(N)
    adj_matrix = generate_adj_matrix(N, agent_activation_probabilities, agent_opinions, r, m, beta)

    # Calculate opinion changes
    for i in range(N):
        dx_dt[i] = opinion_change(i, K, N, adj_matrix, alpha, agent_opinions)

    # Update opinions
    for i in range(N):
        agent_opinions[i] += dt*dx_dt[i]
    
    return(agent_opinions)

def opinion_change(x_index, K, N, adj_matrix, alpha, agent_opinions):
    x = agent_opinions[x_index]
    total_sum = 0
    for j in range(N):
        total_sum += adj_matrix[x_index, j]*math.tanh(alpha*agent_opinions[j])
    opinion_change = -x + K*total_sum
    return(opinion_change)

def generate_adj_matrix(N, agent_activation_probabilities, agent_opinions, r, m, beta):
    # Read this matrix as follows: [i][j] = 1 if agent j influences agent i
    adjacency_matrix = np.zeros((N,N))

    activation_randoms = np.random.rand(N)
    
    for i in range(N):
        # If the agent is active
        if activation_randoms[i] < agent_activation_probabilities[i]:
            # Calculate denominator of equation 3 TODO: NP OPTIMIZE
            total_sum = 0
            for j in range(N):
                # Not necessary
                if j == i:
                    continue
                total_sum += (abs(agent_opinions[i] - agent_opinions[j]) + 1e-9)**-beta

            agents_influened = 0
            while agents_influened <= m:
                agent_index = random.randint(0, N-1)
                influence_prob = abs(agent_opinions[i] - agent_opinions[agent_index])**-beta / total_sum

                # If agent i succesfully influences agent 'agent_index'
                if np.random.rand() < influence_prob:
                    adjacency_matrix[agent_index, i] = 1
                    agents_influened += 1

                    # Reciprocal influence
                    if np.random.rand() < r:
                        adjacency_matrix[i, agent_index] = 1
                        
    return(adjacency_matrix)

def initialize(N, epsilon, gamma):
    # Calculate agent activation probabilities using inverse transform sampling
    # https://en.wikipedia.org/wiki/Inverse_transform_sampling
    # https://en.wikipedia.org/wiki/Cumulative_distribution_function
    # https://en.wikipedia.org/wiki/Inverse_function
    U = np.random.rand(N)
    activation_probabilities = (U * (1 - epsilon**(1-gamma) + epsilon**(1-gamma)))**(1 / (1 - gamma))

    # Initialize agent opinions
    opinions = np.linspace(-1,1,N)
    return activation_probabilities, opinions

def simulate(N, K, r, m, beta, alpha, epsilon, gamma, time_steps, dt):
    agent_activation_probabilities, agent_opinions = initialize(N, epsilon, gamma)

    history = [agent_opinions.copy()]
    for i in range(time_steps):
        agent_opinions = step(N, agent_activation_probabilities, agent_opinions, r, m, beta, K, alpha, dt)

        history.append(agent_opinions.copy())
    
    return np.array(history)

N = 100
time_steps = 10
dt = 0.01

history = simulate(N, K = 3, r = 0.5, m = 10, beta = 3, alpha = 3, epsilon = 0.01, gamma = 2.1, time_steps = time_steps, dt = dt)

plt.plot(history)
plt.xlabel('Time Steps')
#plt.xlim(0,5)
plt.ylabel('Agent Opinions')
plt.title('Opinion Dynamics Simulation')
plt.show()
# plt.figure(figsize=(10, 6))


