# This will be the main python file
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def step(N, agent_activation_probabilities, agent_opinions, r, m, beta, K, alpha, dt):
    # Calculate adjacency matrix based on current opinions and activation probabilities
    adj_matrix = generate_adj_matrix(N, agent_activation_probabilities, agent_opinions, r, m, beta)
    
    # Create new array with opinions after applying tanh function
    opinions_tanh = np.tanh(alpha * agent_opinions)

    # Calculate total sums for each agent using matrix multiplication, summation part of equation 1
    total_sums = adj_matrix @ opinions_tanh

    # Calculate opnion changes, complete equation 1
    opinion_changes = -agent_opinions + K * total_sums

    # Update opinions
    agent_opinions += dt * opinion_changes

    return agent_opinions

def generate_adj_matrix(N, agent_activation_probabilities, agent_opinions, r, m, beta):
    # Read this matrix as follows: [i][j] = 1 if agent j influences agent i
    adjacency_matrix = np.zeros((N,N))

    activation_randoms = np.random.rand(N)
    
    for i in range(N):
        # If the agent is active
        if activation_randoms[i] < agent_activation_probabilities[i]:
            # Calculate distances between agent i and all other agents, add small value to avoid division by zero
            distances = np.abs(agent_opinions[i] - agent_opinions) + 1e-10  

            # Calculate weights of agents (numerator of equation 3)
            weights = distances**-beta

            # Agent i does not influence themself
            weights[i] = 0

            # Influence probabilities
            influence_probabilities = weights / np.sum(weights)

            # Select influenced agents
            influenced_agents = np.random.choice(N, size = m, replace = False, p = influence_probabilities)

            # Apply changes to adjacency matrix for influenced agents
            for agent in influenced_agents:
                adjacency_matrix[agent, i] = 1

                # Reciprocal influence
                if np.random.rand() < r:
                    adjacency_matrix[i, agent] = 1
                        
    return(adjacency_matrix)

def initialize(N, epsilon, gamma):
    # Calculate agent activation probabilities using inverse transform sampling
    # https://en.wikipedia.org/wiki/Inverse_transform_sampling
    # https://en.wikipedia.org/wiki/Cumulative_distribution_function
    # https://en.wikipedia.org/wiki/Inverse_function
    U = np.random.rand(N)
    activation_probabilities = (U * (1 - epsilon**(1-gamma)) + epsilon**(1-gamma))**(1 / (1 - gamma))

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
time_steps = 1000
dt = 0.01

history = simulate(N, K = 3, r = 0.5, m = 10, beta = 0, alpha = 3, epsilon = 0.01, gamma = 2.1, time_steps = time_steps, dt = dt)

time_array = np.arange(time_steps + 1) * dt
plt.plot(time_array, history, alpha=0.3, linewidth=1)

plt.xlabel('Time Steps')
plt.ylabel('Agent Opinions')
plt.title('Opinion Dynamics Simulation')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# plt.figure(figsize=(10, 6))


