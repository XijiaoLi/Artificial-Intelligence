!pip install pgmpy
from pgmpy.utils import get_example_model
child_model = get_example_model('child')
NODES = ['BirthAsphyxia', 'Disease', 'Sick', 'DuctFlow', 'CardiacMixing', 
         'LungParench', 'LungFlow', 'LVH', 'Age', 'Grunting', 'HypDistrib',
         'HypoxiaInO2', 'CO2', 'ChestXray', 'LVHreport', 'GruntingReport',
         'LowerBodyO2', 'RUQO2', 'CO2Report', 'XrayReport']

import numpy.random as npr

def prior_sample(model, nodes):
  """
  Generates and returns a single sample as a {variable: value} dictionary
  """
  sample = {}
  for n in nodes:
    cpd = model.get_cpds(n)
    values = (cpd.state_names)[n]
    probs = prob_given_parents(cpd, sample)
    sample[n] = npr.choice(values, p=probs)
  return sample


def prob_given_parents(cpd, sample):
  """
  Returns probability distribution of the node to which cpd corresponds, 
  conditioned on its parents' values in sample
  """
  factors = cpd.variables
  states = cpd.state_names 

  col = 0
  skip = 1
  for i in range(len(factors)-1):
    parent_var = factors[-i-1]
    parent_val = sample[parent_var]
    ind = states[parent_var].index(parent_val)
    col += ind*skip
    skip *= len(states[parent_var])

  return (cpd.get_values())[:,col]

import numpy.random as npr

def prior_sample(model, nodes):
  """
  Generates and returns a single sample as a {variable: value} dictionary
  """
  sample = {}
  for n in nodes:
    cpd = model.get_cpds(n)
    values = (cpd.state_names)[n]
    probs = prob_given_parents(cpd, sample)
    sample[n] = npr.choice(values, p=probs)
  return sample


def prob_given_parents(cpd, sample):
  """
  Returns probability distribution of the node to which cpd corresponds, 
  conditioned on its parents' values in sample
  """
  factors = cpd.variables
  states = cpd.state_names 

  col = 0
  skip = 1
  for i in range(len(factors)-1):
    parent_var = factors[-i-1]
    parent_val = sample[parent_var]
    ind = states[parent_var].index(parent_val)
    col += ind*skip
    skip *= len(states[parent_var])

  return (cpd.get_values())[:,col]  

def rejection_sample(model, nodes, N, query, evidence={}):
  """
  INPUTS: Problem model, nodes, total number of samples to try (including 
  inconsistent ones), query variable, and a {variable: value} evidence dict
  OUTPUTS: dist, a dictionary providing the estimated distribution for query
           num_consistent, number of consistent samples generated
  """
  dist = {val:0 for val in model.get_cpds(query).state_names[query]}

  # YOUR CODE HERE
  num_samples = 0
  for i in range(N):
    sample = prior_sample(model, nodes) 
    valid = True
    for node in evidence:
      if node in sample and evidence[node] != sample[node]:
        valid = False
        break
    if valid:
      num_samples += 1
      v = sample[query]
      dist[v] += 1
        
  for k in dist:
     dist[k] /= num_samples  
  
  return dist , num_samples

dist, num_samples = rejection_sample(child_model, NODES, 20000, 'XrayReport', evidence={'BirthAsphyxia':'yes'})
print("XrayReport given BirthAsphyxia = yes: ", dist)
print("Number of consistent samples: ", num_samples)

true_pr = {'Normal': 0.2507, 'Oligaemic': 0.2885, 'Plethoric': 0.2069, 'Grd_Glass': 0.0867, 'Asy/Patchy': 0.1672}
diffs = {v: (true_pr[v]-dist.get(v, 0))/true_pr[v] for v in true_pr}
print("Percentage difference: ", diffs)
print("Avg percentage difference: ", sum(diffs.values())/5)

dist, num_samples = rejection_sample(child_model, NODES, 1000, 'XrayReport', evidence={'BirthAsphyxia':'yes'})
diffs = {v: (true_pr[v]-dist.get(v, 0))/true_pr[v] for v in true_pr}
print("----- 1000 requested samples -----")
print("XrayReport given BirthAsphyxia = yes: ", dist)
print("Number of consistent samples: ", num_samples)
print("Percentage difference: ", diffs)
print("Avg percentage difference: ", sum(diffs.values())/5)


def weighted_sample(model, nodes, evidence):
  """
  Generate a sample consistent with evidence, along with corresponding weight
  """
  w = 1
  sample = {}
  for n in nodes:
    cpd = model.get_cpds(n)
    values = (cpd.state_names)[n]
    if n in evidence:
      # YOUR CODE HERE
      sample[n] = evidence[n]      
      probs = prob_given_parents(cpd, sample)      
      i = values.index(sample[n])
      w *= probs[i]

    else:
      probs = prob_given_parents(cpd, sample)
      sample[n] = npr.choice(values, p=probs)
  return sample, w


def likelihood_weighting(model, nodes, N, query, evidence={}):
  """
  INPUTS: Problem model, nodes, query variable, number of samples, 
          and a {variable: value} evidence dict
  OUTPUT: dist, a dictionary providing the estimated distribution for query
  """
  dist = {val:0 for val in model.get_cpds(query).state_names[query]}

  # YOUR CODE HERE
  for i in range(N):
    sample, w = weighted_sample(model, nodes, evidence)
    v = sample[query]
    dist[v] += w
  
  total_w = sum(dist.values())
  for k in dist:
     dist[k] /= total_w  

  return dist


dist = likelihood_weighting(child_model, NODES, 380, 'XrayReport', evidence={'BirthAsphyxia':'yes'})
print("XrayReport given BirthAsphyxia = yes: ", dist)

# CHANGE THE NUMBER OF SAMPLES BELOW
dist = likelihood_weighting(child_model, NODES, 380, 'Disease', evidence={'GruntingReport':'yes', 'LowerBodyO2':'<5', 'RUQO2':'<5', 'CO2Report':'<7.5', 'XrayReport':'Oligaemic'})
print("Disease given reported evidence: ", dist)


for n in range (300, 400, 10):
  count = 0
  for i in range(20):
    dist = likelihood_weighting(child_model, NODES, n, 'XrayReport', evidence={'BirthAsphyxia':'yes'})
    diffs = {v: (true_pr[v]-dist[v])/true_pr[v] for v in true_pr}
    if abs(sum(diffs.values())/5) <= 0.05:
      count += 1
  print(n, count)

