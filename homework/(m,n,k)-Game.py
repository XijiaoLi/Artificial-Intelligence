import numpy as np

def actions(state):
  # Returns indices of all blank spaces on board
  return [i for i,s in np.ndenumerate(state) if s=='.']

def result(state, player, action):
  # Returns a new state (deepcopied) with action space taken by player
  new_state = state.copy()
  new_state[action] = player
  return new_state

def terminal(state, k):
  # Test whether state is a terminal or not; also return game score if yes
  X_indices = [i for i,s in np.ndenumerate(state) if s=='X']
  if k_in_row(X_indices, k): 
    return True, 1
  O_indices = [i for i,s in np.ndenumerate(state) if s=='O']
  blanks = np.count_nonzero(state == '.')
  if k_in_row(O_indices, k) or blanks == 0: 
    return True, -1
  return False, None

#-------------------------------------------------------------------------------
# Utility functions used by terminal (above)

def k_in_row(indices, k):
  # Test whether there are k consecutive indices in a row in the given list
  index_set = set(indices)
  for i in indices:
    for seq in sequences(i, k):
      if seq.issubset(index_set):
        return True
  return False

def sequences(i, k):
  # Return 4 sets of k indices in the "rows" starting from index i
  across = set([(i[0], i[1]+j) for j in range(k)])
  down = set([(i[0]+j, i[1]) for j in range(k)])
  diagdown = set([(i[0]+j, i[1]+j) for j in range(k)])
  diagup = set([(i[0]+j, i[1]-j) for j in range(k)])
  return across, down, diagdown, diagup

def alpha_beta_search(state, player, k):
  # Initialize a game tree search for (m,n,k) game
  # X is maximizing player, O is minimizing player
  if player == 'X':
    value, move = max_value(state, -float("inf"), float("inf"), k)
  else:
    value, move = min_value(state, -float("inf"), float("inf"), k)
  return value, move

def max_value(state, alpha, beta, k):
  isTerminal, score = terminal(state, k)
  if isTerminal:
    return score, None
  # YOUR CODE HERE
  v = -float("inf")
  acts = actions(state)
  for a in acts:
    new_state = result(state, 'X', a)
    v2, a2 = min_value(new_state, alpha, beta, k)
    if v2 > v:
      v, move = v2, a
      alpha = max(alpha, v)
    if v >= beta:
      return v, move
  return v, move

def min_value(state, alpha, beta, k):
  isTerminal, score = terminal(state, k)
  if isTerminal:
    return score, None
  # YOUR CODE HERE
  v = float("inf")
  acts = actions(state)
  for a in acts:
    new_state = result(state, 'O', a)
    v2, a2 = max_value(new_state, alpha, beta, k)
    if v2 < v:
      v, move = v2, a
      beta = min(beta, v)
    if v <= alpha:
      return v, move
  return v, move

def game_loop(state, k, search, X_params=[], O_params=[]):
  # Play a (m,n,k) game using provided search function and parameters
  player = 'X'
  isTerminal = False
  while not isTerminal:
    if player == 'X':
      value, move = search(state, player, k, *X_params)
      state = result(state, player, move)
      player = 'O'
    else:
      value, move = search(state, player, k, *O_params)
      state = result(state, player, move)
      player = 'X'
    print(np.matrix(state), "\n")
    isTerminal, _ = terminal(state, k)

  if value > 0: print("X wins!")
  elif value < 0: print("O wins!")
  else: print("Draw!")

def test1():
  # m, n, k = 3, 3, 3 
  # m, n, k = 2, 5, 3
  # m, n, k  = 3, 4, 3
  m, n, k = 3, 4, 4
  print((m,n,k))
  initial = np.full((m,n), '.')
  game_loop(initial, k, alpha_beta_search)

#------------------------------------------------------------------------

def eval(state, k):
  X_indices = [i for i,s in np.ndenumerate(state) if s=='X']
  O_indices = [i for i,s in np.ndenumerate(state) if s=='O']
  blanks = [i for i,s in np.ndenumerate(state) if s=='.']

  X_and_blanks = X_indices + blanks
  Xset = set(X_indices)
  Xbset = set(X_and_blanks)
  X_score = 0
  for i in X_and_blanks:
    for seq in sequences(i, k):
      if seq.issubset(Xbset):
        ratio = len(seq & Xset)/k
        X_score = max(X_score, ratio)

  O_and_blanks = O_indices + blanks
  Oset = set(O_indices)
  Obset = set(O_and_blanks)
  O_score = 0
  for i in O_and_blanks:
    for seq in sequences(i, k):
      if seq.issubset(Obset):
        ratio = len(seq & Oset)/k
        O_score = max(O_score, ratio)
  
  return X_score - O_score

def alpha_beta_depth_search(state, player, k, max_depth):
  if player == 'X':
    value, move = max_value_depth(state, -float("inf"), float("inf"), k, 1, max_depth)
  else:
    value, move = min_value_depth(state, -float("inf"), float("inf"), k, 1, max_depth)
  return value, move

def max_value_depth(state, alpha, beta, k, depth, max_depth):
  isTerminal, score = terminal(state, k)
  if isTerminal:
    return score, None
  # YOUR CODE HERE
  if depth >= max_depth:
    return eval(state, k), None
  v = -float("inf")
  acts = actions(state)
  for a in acts:
    new_state = result(state, 'X', a)
    v2, a2 = min_value_depth(new_state, alpha, beta, k, depth + 1, max_depth)
    if v2 > v:
      v, move = v2, a
      alpha = max(alpha, v)
    if v >= beta:
      return v, move
  return v, move

def min_value_depth(state, alpha, beta, k, depth, max_depth):
  isTerminal, score = terminal(state, k)
  if isTerminal:
    return score, None
  # YOUR CODE HERE
  if depth >= max_depth:
    return eval(state, k), None
  v = float("inf")
  acts = actions(state)
  for a in acts:
    new_state = result(state, 'O', a)
    v2, a2 = max_value_depth(new_state, alpha, beta, k, depth + 1, max_depth)
    if v2 < v:
      v, move = v2, a
      beta = min(beta, v)
    if v <= alpha:
      return v, move
  return v, move

def test2():
  # m, n, k = 4, 4, 3
  m, n, k = 5, 5, 4
  print((m,n,k))
  initial = np.full((m,n), '.')
  max_depth_X = 5
  max_depth_O = 2
  game_loop(initial, k, alpha_beta_depth_search, [max_depth_X], [max_depth_O])

