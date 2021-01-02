import numpy as np
from matplotlib import pyplot as plt

def actions(state):
  if state == (3,11): return ['G']
  if state[0] == 3 and 0 < state[1] < 11: return ['C']
  return ['<', '>', '^', 'v']

def Qvalue(state, action, values, p, gamma):
  """
  Compute the Q-value for the given state-action pair,
  given a set of values for the problem, with successful transition
  probability p and discount factor gamma.
  """
  i,j = state
  gV = gamma*values
  pn = (1-p)/2

  # Handle goal and cliff states
  if action == 'G':
    return 0
  if action == 'C':
    return -100 + gV[(3,0)]

  # All possible successor states
  left = (i,max(j-1,0))
  right = (i,min(j+1,11))
  up = (max(i-1,0),j)
  down = (min(i+1,3),j)

  # Q-value computation
  if action == '<':
    return p*(-1+gV[left]) + pn*(-1+gV[up]) + pn*(-1+gV[down])
  elif action == '>':
    return p*(-1+gV[right]) + pn*(-1+gV[up]) + pn*(-1+gV[down])
  elif action == '^':
    return p*(-1+gV[up]) + pn*(-1+gV[left]) + pn*(-1+gV[right])
  else:
    return p*(-1+gV[down]) + pn*(-1+gV[left]) + pn*(-1+gV[right])


def value_iteration(values, p, gamma, threshold=1e-6):
  """
  INPUTS: An initial 2D Numpy array of state values, p and gamma parameters, 
  and stopping threshold for value iteration
  OUTPUTS: Converged 2D Numpy array of state values, list of max diffs between
  successive iterations
  """
  max_diffs = [float("inf")]
  while max_diffs[-1] >= threshold:
    # YOUR CODE HERE
    new_values = np.zeros((4,12))
    max_diff = -float("inf")
    for i in range(4):
      for j in range(12):
        best_value = -float("inf")
        for a in actions((i,j)):
          new_value = Qvalue((i,j), a, values, p, gamma)
          if new_value > best_value:
            best_value = new_value
        new_values[(i,j)] = best_value
        diff = abs(best_value - values[i][j])
        if diff > max_diff:
          max_diff = diff
    max_diffs += max_diff,
    values = new_values

  return values, max_diffs

def extract_policy(values, p, gamma):
  # Extract the optimal policy associated with the given optimal values
  policy = np.empty(values.shape, dtype=object)
  for i in range(4):
    for j in range(12):
      best_value = -float("inf")
      for a in actions((i,j)):
        new_value = Qvalue((i,j), a, values, p, gamma)
        if new_value > best_value:
          best_value = new_value
          policy[i,j] = a
  return policy

def solve_cliffworld(p, gamma):
  # Find and show the optimal values and policy for the given parameters
  values, diffs = value_iteration(np.zeros((4,12)), p, gamma)
  print(diffs[2:])
  policy = extract_policy(values, p, gamma) 

  np.set_printoptions(linewidth=100)
  print(np.round(values,2),"\n")
  print(policy,"\n")
  plt.plot(np.arange(1,len(diffs)-1), diffs[2:])
  plt.title("Max difference in values in each iteration")

def test1():
  p = 0.5
  gamma = 0.7
  solve_cliffworld(p, gamma)

#------------------------------------------------------------------------

import random

def epsilon_greedy_action(Qvalues, state, epsilon):
  # Explore a random action from state with probability epsilon
  # Otherwise, greedily choose the best action
  # Qvalues is a dictionary that looks like {(state, action): q-value}
  
  # YOUR CODE HERE
  qmax = -float("inf")
  amax = None
  for sa in Qvalues:
    if sa[0] == state and Qvalues[sa] > qmax:
      qmax = Qvalues[sa]
      amax = sa[1]

  if random.random() < epsilon:
    act = random.choice(actions(state))
    return act
  else:
    return amax

def step(state, action, p):
  # Return successor state and reward upon taking action from state
  i,j = state
  if action == 'C':
    return (3,0), -100

  if action == '<':
    if random.random() < p: return (i,max(j-1,0)), -1
    else: return random.choice([(max(i-1,0),j), (min(i+1,3),j)]), -1
  if action == '>':
    if random.random() < p: return (i,min(j+1,11)), -1
    else: return random.choice([(max(i-1,0),j), (min(i+1,3),j)]), -1
  if action == '^':
    if random.random() < p: return (max(i-1,0),j), -1
    else: return random.choice([(i,max(j-1,0)), (i,min(j+1,11))]), -1
  else:
    if random.random() < p: return (min(i+1,3),j), -1
    else: return random.choice([(i,max(j-1,0)), (i,min(j+1,11))]), -1

def SARSA(Qvalues, p, gamma, alpha, epsilon, episodes=50000):
  # SARSA temporal difference learning using initial Qvalues and given parameters
  # Returns a learned policy (numpy 2d array)
  for i in range(episodes):
    state = (3,0)
    action = epsilon_greedy_action(Qvalues, state, epsilon)
    while state != (3,11):
      next_state, reward = step(state, action, p)
      next_action = epsilon_greedy_action(Qvalues, next_state, epsilon)
      target = Qvalues[(next_state, next_action)]
      Qvalues[(state, action)] += alpha * (reward + gamma*target - Qvalues[(state, action)])
      state = next_state
      action = next_action
  policy = extract_policy(Qvalues)
  return policy

def extract_policy(Qvalues):
  # Extract the optimal policy associated with the given Q-values
  policy = np.empty((4,12), dtype=object)
  for i in range(4):
    for j in range(12):
      policy[i,j] = epsilon_greedy_action(Qvalues, (i,j), 0)
  return policy

def Qlearner(Qvalues, p, gamma, alpha, epsilon, episodes=50000):
  # Q-learning using initial Qvalues and given parameters
  # Returns a learned policy (numpy 2d array)  
  # YOUR CODE HERE
  for i in range(episodes):
    state = (3,0)
    action = epsilon_greedy_action(Qvalues, state, epsilon)
    while state != (3,11):
      next_state, reward = step(state, action, p)
      next_action = epsilon_greedy_action(Qvalues, next_state, 0)
      target = Qvalues[(next_state, next_action)]
      Qvalues[(state, action)] += alpha * (reward + gamma*target - Qvalues[(state, action)])
      state = next_state
      action = epsilon_greedy_action(Qvalues, next_state, epsilon)
  policy = extract_policy(Qvalues)
  return policy

def TD_learn(p, gamma, alpha, epsilon):
  Qvalues = {((i,j),a): 0 for i in range(4) for j in range(12) for a in actions((i,j))}
  # print(Qvalues)
  policy = SARSA(Qvalues, p, gamma, alpha, epsilon)
  print("SARSA policy")   
  print(policy,"\n")
  
  Qvalues = {((i,j),a): 0 for i in range(4) for j in range(12) for a in actions((i,j))}
  policy = Qlearner(Qvalues, p, gamma, alpha, epsilon)
  print("Q-learning policy")  
  print(policy)

def test2():
  p = 1
  gamma = 1
  alpha = 0.9
  epsilon = 0.2
  TD_learn(p, gamma, alpha, epsilon)