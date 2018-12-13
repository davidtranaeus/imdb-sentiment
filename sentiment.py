import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from parserr import *

def tf_idf(samples):
  """tf-idf representation"""
  n_cols = samples.shape[1]
  n_samples = samples.shape[0]

  df = dict()
  for i in range(n_cols):
    df[i] = len(np.nonzero(samples[:,i]))
  
  # doc_len = dict()
  # for i in range(n_samples):
  #   doc_len[i] = np.sum(samples[i])

  for i in range(n_samples):
    for j in range(n_cols):
      samples[i,j] *= (n_samples/df[j])
      # samples[i,j] = (samples[i,j]/doc_len[i]) * np.log(n_samples/df[j])

  return samples

def binary(samples):
  """one hot encoding representation"""
  return (samples > 0).astype(np.int)

def prepare_data(samples, targets, is_tfidf=False, is_binary=False, is_lsa=False, is_norm=False):
  targets = np.ravel(targets)

  if is_binary:
    print('Setting binary values')
    samples = binary(samples)
  
  if is_tfidf:
    print('Setting tf-idf values')
    samples = tf_idf(samples)

  if is_lsa:
    print('Setting LSA')
    svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    svd.fit(samples)
    samples = svd.fit_transform(samples)

  if is_norm:
    print('Normalizing')
    for i in range(len(samples)):
      samples[i] /= np.linalg.norm(samples[i])


  return samples, targets
  
if __name__ == "__main__":
  classifier = 'regression'

  is_binary = False
  is_tfidf = False
  is_lsa = False
  is_norm = False

  p_size = 1000

  print('Reading training data')
  train_samples, train_targets_data, vocabulary = trainData(crop=True, Sample_Size=p_size)
  train_samples = train_samples[:-2]
  train_targets_data = train_targets_data[:-2]
  train_targets = train_targets_data[:,1].reshape(-1,1)
  train_samples, train_targets = prepare_data(
    train_samples, 
    train_targets,
    is_tfidf=is_tfidf,
    is_binary=is_binary,
    is_lsa = is_lsa)
  print(train_samples.shape)

  print('Reading test data')
  test_samples, test_targets_data = testData(vocab = vocabulary, crop=True, Sample_Size=p_size)
  test_samples = test_samples[:-2]
  test_targets_data = test_targets_data[:-2]
  test_targets = test_targets_data[:,1].reshape(-1,1)
  test_samples, test_targets = prepare_data(
    test_samples, 
    test_targets,
    is_tfidf=is_tfidf,
    is_binary=is_binary,
    is_lsa = is_lsa)

  if classifier == 'regression':
    print('Setting up logistic regression')
    model = LogisticRegression(solver='lbfgs', max_iter=500) 
  elif classifier == 'knn':
    print('Setting up KNN')
    model = KNeighborsClassifier(n_neighbors=3)
  elif classifier == 'mlp':
    print("Setting up MLP")
    model = MLPClassifier(hidden_layer_sizes=(50,30,))
  elif classifier == 'svm':
    print('Setting up SVM')
    model = svm.SVC(kernel='rbf', gamma='auto')


  print('Training model')
  model.fit(train_samples, train_targets)

  print('Accuracy: ', end='')
  print(model.score(test_samples, test_targets))

  print(confusion_matrix(
    test_targets,
    model.predict(test_samples)))

