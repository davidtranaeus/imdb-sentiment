import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from parser import *
import argparse

def tf_idf(samples):
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
  return (samples > 0).astype(np.int)

def prepare_data(samples, targets, is_tfidf=False, is_binary=False):
  """Prepares data if tf-idf or binary representation is selected"""
  targets = np.ravel(targets)

  if is_binary:
    print('Setting binary values')
    samples = binary(samples)
  
  if is_tfidf:
    print('Setting tf-idf values')
    samples = tf_idf(samples)

  return samples, targets
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--test', help='Select test set "books" or "kitchen"')
  parser.add_argument('--classifier', help='Select classifier "regression" or "mlp"')
  parser.add_argument('-t', '--tfidf', action='store_true', help='Use tfidf data representation')
  parser.add_argument('-b', '--binary', action='store_true', help='Use binary data representation')
  args = parser.parse_args()
 
  if (args.test != 'books' and args.test != 'kitchen'):
    parser.error('No test set was requested. Select test set by setting --test to either "books" or "kitchen"')
  if (args.classifier != 'regression' and args.classifier != 'mlp'):
    parser.error('No classifier was set. Select classifier set by setting --c to either "regression" or "mlp"')

  classifier = args.classifier
  is_binary = True if args.binary else False
  is_tfidf = True if args.tfidf else False
  
  if args.test == 'books':
    test_directories = [
      'amazon_reviews/book_reviews/neg',
      'amazon_reviews/book_reviews/pos']
  if args.test == 'kitchen':
    test_directories = [
      'amazon_reviews/kitchen_reviews/neg',
      'amazon_reviews/kitchen_reviews/pos']

  p_size = 2000

  print('Creating document-term matrix for training and validation set')
  train_samples, train_targets_data, vocabulary = train_data(crop=True, sample_Size=p_size)
  train_samples = train_samples[:-2]
  train_targets_data = train_targets_data[:-2]
  train_targets = train_targets_data[:,1].reshape(-1,1)
  train_samples, train_targets = prepare_data(
    train_samples, 
    train_targets,
    is_tfidf=is_tfidf,
    is_binary=is_binary)

  print('Creating document-term matrix for testing set')
  test_samples, test_targets_data = test_data(dir_names=test_directories, vocab = vocabulary, crop=True, sample_Size=1000)
  test_samples = test_samples[:-2]
  test_targets_data = test_targets_data[:-2]
  test_targets = test_targets_data[:,1].reshape(-1,1)
  test_samples, test_targets = prepare_data(
    test_samples, 
    test_targets,
    is_tfidf=is_tfidf,
    is_binary=is_binary)

  if classifier == 'regression':
    print('\nSetting up model for logistic regression')
    model = LogisticRegression(solver='lbfgs', max_iter=500) 
  if classifier == 'mlp':
    print('\nSetting up model for multilayer perceptron')
    model = MLPClassifier(hidden_layer_sizes=(50,30,))

  print('\nComputing validation accuracy with 10fCV')
  kf = KFold(n_splits=10)
  kf.get_n_splits(train_samples)

  k_fold_scores = []
  val_confusion = np.zeros((2,2))
  for train_index, test_index in kf.split(train_samples):
    X_train, X_test = train_samples[train_index], train_samples[test_index]
    y_train, y_test = train_targets[train_index], train_targets[test_index]
    model.fit(X_train, y_train)
    k_fold_scores.append(model.score(X_test, y_test))
    val_confusion += confusion_matrix(y_test, model.predict(X_test))

  print('Average model accuracy from 10fCV: {:0.3f}'.format(np.average(k_fold_scores)))
  print('\nConfusion matrix:')
  print((val_confusion/10).astype(np.int))

  print('\nTesting model accuracy on test set')
  run_times = 10
  test_scores = []
  confusion = np.zeros((2,2))
  for i in range(run_times):
    model.fit(train_samples, train_targets)
    test_scores.append(model.score(test_samples, test_targets))
    confusion += confusion_matrix(test_targets, model.predict(test_samples))
  
  print('Average model accuracy on test set: {:0.3f}'.format(np.average(test_scores)))
  print('\nConfusion matrix:')
  print((confusion/run_times).astype(np.int))

  if classifier == 'regression':
    sorted_idx = np.argsort(model.coef_[0])
    print('\nThe words with greatest impact for classifying a review as negative:')
    for key, value in vocabulary.items():
      if value in sorted_idx[:15]:
        print(key, end=' ')
    
    print('\n\nThe words with greatest impact for classifying a review as positive:')
    for key, value in vocabulary.items():
      if value in sorted_idx[-15:]:
        print(key, end=' ')


