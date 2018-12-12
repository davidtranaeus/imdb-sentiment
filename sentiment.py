import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPRegressor
from parserr import *

def tf_idf(samples):
  """tf-idf representation"""
  n_cols = samples.shape[1]
  n_samples = samples.shape[0]

  df = dict()
  for i in range(n_cols):
    df[i] = len(np.nonzero(samples[:,i]))

  for i in range(n_samples):
    for j in range(n_cols):
      samples[i,j] *= (n_samples/df[j])

  return samples

def binary(samples):
  """one hot encoding representation"""
  return (samples > 0).astype(np.int)

def less_data(data, n_of_each = 5000):
  n_samples = data.shape[0]
  splt_point = int(n_samples/2)
  sample_dim = data.shape[1]
  
  less = np.zeros((n_of_each*2, sample_dim))
  less[:n_of_each] = data[:n_of_each]
  less[n_of_each:] = data[splt_point:splt_point+n_of_each]

  return less

def prepare_data(samples, targets, is_tfidf=False, is_binary=False):
  if is_binary:
    print('Setting binary values')
    samples = binary(samples)
  
  if is_tfidf:
    print('Setting tf-idf values')
    samples = tf_idf(samples)

  return samples, targets
  
if __name__ == "__main__":
  use_knn = False
  use_regressor = False
  is_binary = False
  is_tfidf = False

  do_crop = True
  partition_size = 1000

  print('Reading training data')
  train_samples, train_targets_data, vocabulary = trainData(crop=do_crop, Sample_Size=partition_size)
  train_samples = train_samples[:-2]
  train_targets = train_targets_data[:,1].reshape(-1,1)
  train_samples, train_targets = prepare_data(
    train_samples, 
    train_targets,
    is_tfidf=is_tfidf,
    is_binary=is_binary)

  if use_regressor:
    print('Training logistic regression')
    model = LogisticRegression() 
  elif use_knn:
    print('Training KNN')
    model = KNeighborsClassifier(n_neighbors=3)
  else:
    print("Training MLP")
    model = MLPRegressor(hidden_layer_sizes=(50,30,), max_iter=200)

  model.fit(train_samples, train_targets)

  print('Reading test data')
  test_samples, test_targets_data = testData(vocab = vocabulary, crop=do_crop, Sample_Size=partition_size)
  test_samples = test_samples[:-2]
  test_targets = test_targets_data[:,1].reshape(-1,1)
  test_samples, test_targets = prepare_data(
    test_samples, 
    test_targets,
    is_tfidf=is_tfidf,
    is_binary=is_binary)

  print('Accuracy: ', end='')
  print(model.score(test_samples, test_targets))

  print(confusion_matrix(
    test_targets,
    model.predict(test_samples)))