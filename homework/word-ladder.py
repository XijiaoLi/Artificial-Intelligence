"""
In this assignment you will implement and use search algorithms to solve word ladder puzzles.
Given two English words, the goal is to transform the first word into the second word by changing
one letter at a time. The catch is that each new word in the process must also be an English 
(dictionary) word. The following function encodes this process using the pyenchant package.
"""

# RUN THIS ONCE IN THE BEGINNING TO INSTALL PYENCHANT
# !pip install pyenchant
# !apt-get install libenchant1c2a

import enchant, string

def successors(state):
  """
  Given a word, find all possible English word results from changing one letter.
  Return a list of (action, word) pairs, where action is the index of the
  changed letter.
  """
  d = enchant.Dict("en_US")
  child_states = []
  for i in range(len(state)):
    new = [state[:i]+x+state[i+1:] for x in string.ascii_lowercase]
    words = [x for x in new if d.check(x) and x != state]
    child_states = child_states + [(i, word) for word in words]
  return child_states


"""
Make sure you understand the description above, as well as the partial implementation given
to you below. Then complete the loop portion of best_first_search where indicated. Some hints:
1. The goal test can be done by checking string equality with the goal state.
2. Remember to appropriately update max_frontier and nodes_expanded whenever one of these quantities changes.
3. Return the solution along with the max frontier size and number of nodes expanded.
4. Use the provided expand function for node expansion.
5. Push tuples into the frontier queue in the same way as in the initialization.
"""
from heapq import heappush, heappop

def best_first_search(state, goal, f):
  """
  Inputs: Initial state, goal state, priority function
  Returns node containing goal or None if no goal found, max frontier size, 
  and total nodes expanded
  """
  node = {'state':state, 'parent':None, 'action':None, 'depth':0, 'cost':0}
  frontier = []
  heappush(frontier, (f(node, goal), id(node), node))
  reached = {state: node}
  max_frontier = 1
  nodes_expanded = 0

  while frontier:
    node = heappop(frontier)[2]
    # YOUR CODE HERE    
    nodes_expanded += 1
    if node['state'] == goal:
      return node, max_frontier, nodes_expanded
    else:
      childern = expand(node)
      for n in childern:
        word = n['state']
        if word in reached and (reached[word])['cost'] <= n['cost']:
          continue
        else:
          heappush(frontier, (f(n, goal), id(n), n))
          reached[word] = n
      max_frontier = max(max_frontier, len(frontier))

  return None, max_frontier, nodes_expanded

def expand(node):
  """
  Given a node, return a list of successor nodes
  """
  vowels = ['a', 'e', 'i', 'o', 'u']
  state = node['state']
  children = []
  for successor in successors(state):
    cost = 2 if state[successor[0]] in vowels else 1
    children.append({'state':successor[1], 'parent':node,
                     'action':successor[0], 'depth':node['depth']+1,
                     'cost':node['cost']+cost})
  return children


"""
The best_first_search implementation is general. Each specific search algorithm 
behavior can be produced by specifying the appropriate priority function f. Complete
the priority functions below for depth-first, breadth-first, and uniform-cost search.
(Although goal is an argument, it will not be used in these functions.)
"""
def f_bfs(node, goal=None):
  # YOUR CODE HERE
  return node['depth']

def f_dfs(node, goal=None):
  # YOUR CODE HERE
  return (-1) * node['depth']

def f_ucs(node, goal=None):
  # YOUR CODE HERE
  return node['cost']


"""
Let's now try our hand at informed search. We will be able to reuse best_first_search once 
again---we just need to implement the priority function, which will now utilize a heuristic. 
A suitable heuristic is the Hamming distance between a state and the goal: the number of 
indices where the corresponding letters are different. We can also make this heuristic more 
sophisticated by adding to the Hamming distance the number of vowels in the current word that
must be changed to reach the goal word.
Complete the f_astar and f_astar_vowel functions below to use the "simple" and "sophisticated"
Hamming distances, respectively, described above as the heuristics. Don't forget to add in 
cumulative node cost as well.
"""
def f_astar(node, goal):
  word = node['state']
  h = 0
  for i in range(len(word)):
    if word[i] != goal[i]:
      h += 1
  return node['cost'] + h

def f_astar_vowel(node, goal):
  # YOUR CODE HERE
  vowels = ['a', 'e', 'i', 'o', 'u']
  word = node['state']
  h = 0
  for i in range(len(word)):
    if word[i] != goal[i]:
      if word[i] in vowels:
        h += 2
      else: 
        h += 1
  return node['cost'] + h


"""
We now have a complete implementation for DFS, BFS, and UCS. The following functions
will help us present the results in a friendly way. The first function puts together
the sequence of words in the solution. The second prints out all the results together.
"""
def sequence(node):
  words = [node['state']]
  while node['parent'] is not None:
    node = node['parent']
    words.insert(0, node['state'])
  return words

def results(solution):
  if solution[0] is not None:
    print(sequence(solution[0]))
    print("Total cost:", solution[0]['cost'])
    print("Max frontier size:", solution[1])
    print("Nodes expanded:", solution[2])
  else: print("No solution found!")
  print("")


"""
Time to test our implementation! We'll start off with a "simple" 3-letter word ladder
to compare DFS and BFS. Try running the first cell below a couple times and take note
of the results. Also feel free to experiment with other 3-letter word ladders (you can
search online for common ones to verify their solutions). Note: Sometimes DFS may get
stuck and continue running for a long time. If that happens, you can kill the process
and try running the cell again.
"""
def main():
	start = 'cat'
	goal = 'cop'

	solution = best_first_search(start, goal, f_dfs)
	print("DFS")
	results(solution)

	solution = best_first_search(start, goal, f_bfs)
	print("BFS")
	results(solution)

	start = 'small'
	goal = 'large'

	solution = best_first_search(start, goal, f_bfs)
	print("BFS")
	results(solution)

	solution = best_first_search(start, goal, f_ucs)
	print("UCS")
	results(solution)

	start = 'small'
	goal = 'large'

	solution = best_first_search(start, goal, f_astar)
	print("A* simple Hamming")
	results(solution)

	solution = best_first_search(start, goal, f_astar_vowel)
	print("A* Hamming with vowels")
	results(solution)
