import numpy as np

def read_sentence(f):
  sentence = []
  while True:
    line = f.readline()
    if not line or line == '\n':
      return sentence
    line = line.strip()
    word, tag = line.split("\t", 1)
    sentence.append((word, tag))

def read_corpus(file):
  f = open(file, 'r', encoding='utf-8')
  sentences = []
  while True:
    sentence = read_sentence(f)
    if sentence == []:
      return sentences
    sentences.append(sentence)

training = read_corpus('train.upos.tsv')
TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 
        'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
NUM_TAGS = len(TAGS)

alpha = 0.1
tag_counts = np.zeros(NUM_TAGS)
transition_counts = np.zeros((NUM_TAGS,NUM_TAGS))
obs_counts = {}

for sent in training:
  for i in range(len(sent)):
    word = sent[i][0]
    pos = TAGS.index(sent[i][1])
    tag_counts[pos] += 1
    if i < len(sent)-1:
      transition_counts[TAGS.index(sent[i+1][1]), pos] += 1
    if word not in obs_counts:
      obs_counts[word] = np.zeros(NUM_TAGS)
    (obs_counts[word])[pos] += 1

TPROBS = transition_counts / np.sum(transition_counts, axis=0)
OPROBS = {'#UNSEEN': alpha*np.ones(NUM_TAGS) / (tag_counts+alpha)}
for word, counts in obs_counts.items():
  OPROBS[word] = np.divide(counts, tag_counts+alpha)

def unigram(obs):
  # Returns the tag of the word obs, as predicted by a unigram model
  # YOUR CODE HERE
  if not obs in OPROBS:
    obs = '#UNSEEN'
  i = np.argmax(OPROBS[obs])
  return TAGS[i]

print(tag_counts)

def evaluate(sentences, method):
  correct = 0
  correct_unseen = 0
  num_words = 0
  num_unseen_words = 0

  for sentence in sentences:
    words = [sent[0] for sent in sentence]
    pos = [sent[1] for sent in sentence]
    unseen = [word not in OPROBS for word in words]
    if method == 'unigram':
      predict = [unigram(w) for w in words]
    elif method == 'viterbi':
      predict = viterbi(words)
    else:
      print("invalid method!")
      return

    if len(predict) != len(pos):
      print("incorrect number of predictions")
      return
    correct += sum(1 for i,j in zip(pos, predict) if i==j)
    correct_unseen += sum(1 for i,j,k in zip(pos, predict, unseen) if i==j and k)
    num_words += len(words)
    num_unseen_words += sum(unseen)
  
  print("Accuracy rate on all words: ", correct/num_words)
  if num_unseen_words > 0:
    print("Accuracy rate on unseen words: ", correct_unseen/num_unseen_words)

print("Training data evaluation")
evaluate(training, 'unigram')
test = read_corpus('test.upos.tsv')
print("")
print("Test data evaluation")
evaluate(test, 'unigram')

def elapse_time(m):
  """
  Given a "message" distribution over tags, return an updated distribution
  after a single timestep using Viterbi update, along with a list of the 
  indices of the most likely prior tag for each current tag
  """
  # mprime = np.zeros(NUM_TAGS)
  # prior_tags = np.zeros(NUM_TAGS, dtype=np.int8)
  #YOUR CODE HERE
  p = np.multiply(m, TPROBS)
  prior_tags = np.argmax(p, axis=1)
  mprime = np.amax(p, axis=1)

  return mprime, prior_tags

m0 = np.ones(NUM_TAGS) /NUM_TAGS
m1_prime, m1_prior_tags = elapse_time(m0)
print(m1_prime)
print(m1_prior_tags)


def observe(mprime, obs):
  """
  Given a "message" distribution over tags, return an updated distribution
  by weighting mprime by the emission probabilities corresponding to obs
  """
  # m = np.zeros(NUM_TAGS)
  # YOUR CODE HERE
  if not obs in OPROBS:
    obs = '#UNSEEN'
  m = np.multiply(OPROBS[obs], mprime)
  return m

m1 = observe(m1_prime, '#UNSEEN')
print(m1)

def viterbi(observations):
  """
  Given a list of word observations, return a list of predicted tags
  """
  m = np.ones(NUM_TAGS)/NUM_TAGS
  # YOUR CODE HERE
  hist = []

  # "Forward" phase of the Viterbi algorithm
  for obs in observations:
    mprime, prio = elapse_time(m)
    hist += prio,
    m = observe(mprime, obs)
  
  # "Backward" phase of the Viterbi algorithm
  i = np.argmax(m)
  ret = [TAGS[i]]
  for j in range(len(hist))[::-1]:
    i = hist[j][i]
    ret += TAGS[i],
  
  return (ret[::-1])[1:]


print(viterbi(['a', 'round', 'circle']))
print(viterbi(['play', 'another', 'round']))
print(viterbi(['walk', 'round', 'the', 'fence']))

print("Training data evaluation")
evaluate(training, 'viterbi')
print("")
print("Test data evaluation")
evaluate(test, 'viterbi')

def hapax_POS_counts(obs_counts):
  """
  Given a dictionary of word to count arrays, return the total number 
  of POS tag appearances only for the words with a total count of 1
  """
  hapax = np.zeros(NUM_TAGS)
  # YOUR CODE HERE
  for obs in obs_counts:
    if np.sum(obs_counts[obs]) == 1:
      hapax = np.add(hapax, np.array(obs_counts[obs]))
  print(np.divide(hapax, np.sum(hapax)))
  return hapax

hapax = hapax_POS_counts(obs_counts)

hapax = hapax_POS_counts(obs_counts)
for word, counts in obs_counts.items():
  OPROBS[word] = np.divide(counts, tag_counts + hapax)
OPROBS['#UNSEEN'] = np.divide(hapax, tag_counts + hapax)

print("Test data evaluation")
evaluate(test, 'viterbi')
