import numpy as np
import pickle
import scipy.io

class Importer(object):
  def __init__(self, dat_file, train_frac):
    with open(dat_file, 'rb') as f:
      data = pickle.load(f)
    # data = scipy.io.loadmat(mat_file)

    print(data.keys())
    self.graphs = data['rs_graphs']
    self.signals = data['rs_signals']
    self.labels = data['rs_labels']

    assert ((self.graphs.shape[0] == self.signals.shape[0]) & (self.signals.shape[0] == self.labels.shape[0]))

    self.data_length = int(self.graphs.shape[0])
    self.train_length = int(np.floor(self.data_length * train_frac))
    self.epoch = int(0)
    self.cursor = int(0)
    self.cursor_test = int(self.train_length)

  def get_dimension(self):
    return self.signals.shape

  def next_batch_train(self, batch_size):
    # Graph: batch_size x dim x dim
    # Signal: batch_size x 1 x dim
    # Labels: batch_size

    if self.cursor+batch_size >= self.train_length:
      graphs_a = self.graphs[self.cursor:self.train_length]
      signals_a = self.signals[self.cursor:self.train_length]
      labels_a = self.labels[self.cursor:self.train_length]

      self.cursor = (self.cursor+batch_size)-self.train_length
      self.epoch += 1

      graphs_b = self.graphs[0:self.cursor]
      signals_b = self.signals[0:self.cursor]
      labels_b = self.labels[0:self.cursor]

      graphs = np.concatenate((graphs_a, graphs_b), axis=0)
      signals = np.concatenate((signals_a, signals_b), axis=0)
      labels = np.concatenate((labels_a, labels_b), axis=0)
    else:
      graphs = self.graphs[self.cursor:self.cursor+batch_size]
      signals = self.signals[self.cursor:self.cursor+batch_size]
      labels = self.labels[self.cursor:self.cursor+batch_size]
      self.cursor += batch_size

    # signals.shape = (n, 32)
    # signals_placeholder.shape = (n, 1, 32)

    # labels.shape = (n, 1)
    # labels_placeholder.shape = (n, )
    signals = signals.reshape(signals.shape[0], 1, signals.shape[1])
    labels = labels.squeeze().astype('int32')

    indices = np.arange(batch_size)
    np.random.shuffle(indices)

    graphs = graphs[indices]
    signals = signals[indices]
    labels = labels[indices]

    return graphs, signals, labels

  def next_batch_test(self, batch_size):
    # Graph: batch_size x dim x dim
    # Signal: batch_size x 1 x dim
    # Labels: batch_size

    if self.cursor_test+batch_size >= self.data_length:
      graphs_a = self.graphs[self.cursor_test:self.data_length]
      signals_a = self.signals[self.cursor_test:self.data_length]
      labels_a = self.labels[self.cursor_test:self.data_length]

      self.cursor_test = (self.cursor_test+batch_size)-self.data_length+self.train_length
      self.epoch += 1

      graphs_b = self.graphs[self.train_length:self.cursor_test]
      signals_b = self.signals[self.train_length:self.cursor_test]
      labels_b = self.labels[self.train_length:self.cursor_test]

      graphs = np.concatenate((graphs_a, graphs_b), axis=0)
      signals = np.concatenate((signals_a, signals_b), axis=0)
      labels = np.concatenate((labels_a, labels_b), axis=0)
    else:
      graphs = self.graphs[self.cursor_test:self.cursor_test+batch_size]
      signals = self.signals[self.cursor_test:self.cursor_test+batch_size]
      labels = self.labels[self.cursor_test:self.cursor_test+batch_size]
      self.cursor_test += batch_size

    signals = signals.reshape(signals.shape[0], 1, signals.shape[1])
    labels = labels.squeeze().astype('int32')

    indices = np.arange(batch_size)
    np.random.shuffle(indices)

    graphs = graphs[indices]
    signals = signals[indices]
    labels = labels[indices]

    return graphs, signals, labels

  def reset(self):
    self.cursor = 0
    self.epoch = 0

