import random
import numpy as np
print("************PART A************")
probability = 0.1
total_states = 91 #At the most 91 possible as we increase the probability linearly. 0.1 state, 0.11 state... 1 state
probability_matrix = np.zeros((total_states,total_states)) #Initialise probability matrix
pi = np.zeros(total_states) #Initialise stationary distribution vector
reward = np.zeros(total_states) #Initialise reward vector
for i in range(total_states):
    probability_matrix[i][0] = probability + i*0.01 #Assigning linearly increasing failure probability for each state
    if i < 90:
        probability_matrix[i][i+1] = 1 - probability - i*0.01 #Probability of going from one state to next state without failure
pi[0] = 1 #For being able to do matrix multiplication
for i in range(total_states): #matrix multiplication to get the stationary distribution
    pi = pi @ probability_matrix

np.set_printoptions(suppress=True)#To get rid of the exponent from the vector
print("Stationary distribution : ", pi)
reward[0] = 1 #For being able to do matrix multiplication
phi = reward @ pi
print("Long run avg cost = ",phi)



#PART B via Monte Carlo SIMULATION
print("\n")
print("**************PART B************")
steps = 100000 #Initialise the number of iterations for the simulation
start_state = 0 #Set start state
pi1 = np.zeros(total_states) #Initialise stationary distribution vector
pi1[start_state] = 1 
prev_state = start_state #Initialise previous state variable
i=0
while i<steps:
    curr_state = np.random.choice(np.arange(0, 91),p=probability_matrix[prev_state]) #Pick the current state based on the probability of previous state
    pi1[curr_state]+=1
    prev_state = curr_state
    i=i+1
print("Stationary distribution via Monte Carlo Simulation: ",pi1/steps)
phi1 = reward @ pi1
print("Long run average cost via Monte Carlo Simulation: ",phi1/steps)


#PART C:
print("******PART C*******")
reward1 = np.append(reward,0) #initialise reward vector
poisson_eqn = np.identity(91)-probability_matrix #Subtract probaability matrix from identity matrix to get the coefficients
poisson_eqn = np.pad(poisson_eqn, [(0, 1), (0, 1)], mode='constant', constant_values=1) #Padding the last row and last column with 1
poisson_eqn[91][91] = 0 #Setting the value to 1 at position 91,91
poisson_eqn = np.linalg.solve(poisson_eqn, reward1) #Solving 91 equations to get long run average
print("Long run avg cost after solving poisson equation:",poisson_eqn[-1])

# PART D 1
print("*************PART D*****************")
probability_matrix1 = probability_matrix
rewards1 = np.zeros((total_states))
for i in range(0,total_states):
    rewards1[i] = i*0.01 + 0.1

probability_matrix2 = np.zeros((total_states,total_states)) # initialising probability and reward matrices for 2 set of actions
rewards2 = np.zeros((total_states))
for i in range(0,total_states):
    probability_matrix2[i][0] = 1
    rewards2[i] = 0.5


policy = np.zeros((total_states)) 
for i in range(total_states): 
    r = np.zeros((total_states))
    P = np.zeros((total_states,total_states))
    for j in range(total_states): 
        if policy[j] == 1: 
            r[j] = rewards1[j]
            P[j] = probability_matrix1[j]
        else:
            r[j] = rewards2[j]
            P[j] = probability_matrix2[j]
  
    
    reward_temp = np.append(r,0)
    poisson_eqn = np.identity(91)-P
    poisson_eqn = np.pad(poisson_eqn, [(0, 1), (0, 1)], mode='constant', constant_values=1)
    poisson_eqn[91][91] = 0
    poisson_eqn_solved = np.linalg.solve(poisson_eqn, reward_temp) # solving poissons equation 
    poisson_without_phi = poisson_eqn_solved[:-1] # solution of poisson equation with the phi value
    actions1 = rewards1 + probability_matrix1@poisson_without_phi
    actions2 = rewards2 + probability_matrix2@poisson_without_phi
    policy = np.zeros((len(actions1))) # optimal policy matrix
    for i in range(0,len(actions1)):
        if actions1[i]<actions2[i]: # checking for optimal action
            policy[i] = 1
        else:
            policy[i] = 2


    if i == 90: # terminate the iteration
        reward1 = np.append(r,0)
        poisson_eqn = np.identity(91)-P
        poisson_eqn = np.pad(poisson_eqn, [(0, 1), (0, 1)], mode='constant', constant_values=1)
        poisson_eqn[91][91] = 0
        poisson_eqn = np.linalg.solve(poisson_eqn, reward1)
        phi = poisson_eqn[-1]

print("Best policy")
print(policy)
print("Phi value from policy iteration - ",phi)



# PART D 2
V= np.zeros((total_states))
print(len(rewards1))
for i in range(total_states):
    V_actions1 = rewards1 + probability_matrix1@V
    V_actions2 = rewards2 + probability_matrix2@V
    V_star = np.minimum(V_actions1,V_actions2) # selecting optimal V
    if i == 90:
        phi = (V_star - V)
    V = V_star
    
phi_final = phi[0]
print("Value Iteration expected limiting reward (Long running average)", phi_final)