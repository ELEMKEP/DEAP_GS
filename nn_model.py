# 2016 Soobeom Jang.
#
# Using tensorflow, making neural network and its training scheme.
#
# Started at 16/07/27.
# 12/15 revision: changed to suitable for TF 0.12
#
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np


def inference_gnn(graphs, signals, output_dim):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    graphs: input graphs. #node of all graphs should be same.
    signals: graph signals corresponds to the graphs.
    output_dim: #output of network. same as #class.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """

  hidden1 = gshiftnn_layer(graphs, signals, 'hidden1')
  hidden2 = gshiftnn_layer(graphs, hidden1, 'hidden2')
  logits = softmax_linear_layer(hidden2, output_dim, 'softmax_linear')

  return logits


def inference_nn(signals, hiddens, output_dim):
  signals_squeeze = tf.squeeze(signals, None, 'signal_squeeze')

  if len(hiddens) > 0:
    for idx in range(len(hiddens)):
      if idx == 0:
        hidden = sigmoid_layer(signals_squeeze, hiddens[0], 'hidden1')
      else:
        hidden = sigmoid_layer(hidden, hiddens[idx], 'hidden'+str(idx))

  logits = softmax_linear_layer(hidden, output_dim, 'softmax_linear')
  return logits


def sigmoid_layer(signal, output_dim, name_scope):
  # ordinary sigmoid neural network layer.
  # Assumption: coords is same for all graphs, although their topologies can be different.
  with tf.name_scope(name_scope):
    input_size = int(str(signal.get_shape()[1]))

    weights = tf.Variable(
      tf.truncated_normal([input_size, output_dim], stddev=1.0/math.sqrt(input_size)),
      name='weights'
    )
    biases = tf.Variable(tf.zeros([output_dim]),
                         name='biases')
    output = tf.nn.relu(tf.matmul(signal, weights) + biases)
  return output

def inference_coord_conv(coords, graphs, signals, output_dim, threshold=None):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    coords: position of graph nodes. n x 2 matrix.
    graphs: input graphs. #node of all graphs should be same.
    signals: graph signals corresponds to the graphs.
    output_dim: #output of network. same as #class.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """

  hidden1 = gcoord_conv_layer(coords, graphs, signals, 'hidden1', threshold)
  hidden2 = gcoord_conv_layer(coords, graphs, hidden1, 'hidden2', threshold)
  logits = softmax_linear_layer(hidden2, output_dim, 'softmax_linear')

  return logits


def gcoord_conv_layer(coords, gA, signals, name_scope, threshold=0.3):
  # Assumption: coords is same for all graphs, although their topologies can be different.
  with tf.name_scope(name_scope):
    assert gA.get_shape()[1] == gA.get_shape()[2], 'gcoord_conv_layer: Graph edge matrix should be a square matrix!'

    if threshold is None:
      threshold = 100

    input_dim = int(str(gA.get_shape()[1]))
    n_records = int(str(gA.get_shape()[0]))

    x = tf.constant(np.reshape(coords[:, 0], [-1, 1]), dtype=tf.float32)
    y = tf.constant(np.reshape(coords[:, 1], [-1, 1]), dtype=tf.float32)

    threshold = tf.constant(threshold, dtype=tf.float32)

    x_left = tf.tile(x, [1, input_dim], name='x_left')

    x_right = tf.transpose(tf.tile(x, [1, input_dim]), name='x_right')

    y_left = tf.tile(y, [1, input_dim], name='y_left')
    y_right = tf.transpose(tf.tile(y, [1, input_dim]), name='y_right')

    dist = tf.sqrt(tf.squared_difference(x_left, x_right) + tf.squared_difference(y_left, y_right), name='dist')

    Dt = tf.cast(tf.less_equal(dist, threshold), tf.float32)

    Dt_expand = tf.tile(tf.reshape(Dt, [1, input_dim, input_dim]), [n_records, 1, 1])

    # Tiling --> A* = A .* thresh(D)
    gAD = tf.mul(gA, Dt_expand, name='coord_A')

  return gshiftnn_layer(gAD, signals, name_scope)


def gshiftnn_layer(gA, signals, name_scope):
  # Graph W - (batch_size, input_dim, input_dim)
  # Signals - (batch_size, 1, input_dim) --> (batch_size, input_dim, input_dim)
  # Network W - (input_dim, input_dim) --> (batch_size, input_dim, input_dim)
  # Biases - (1, input_dim) --> (batch_size, input_dim)
  with tf.name_scope(name_scope):
    assert gA.get_shape()[1] == gA.get_shape()[2], 'gshiftnn_layer: Graph edge matrix should be a square matrix!'

    input_dim = int(str(gA.get_shape()[1]))
    n_records = int(str(gA.get_shape()[0]))

    # Signal expansion
    signals_expand = tf.tile(signals, [1, input_dim, 1],
                             name='signals_expand')

    # Weight definition and symmetrize, expansion
    normal = tf.truncated_normal([input_dim, input_dim],
                          stddev= 1.0 / np.sqrt(float(input_dim)))
    weights = tf.Variable(normal, name='weights')
    weights_symm = tf.truediv(tf.add(weights, tf.transpose(weights)),
                             tf.constant(2.),
                             name='weights_symm')
    weights_expand = tf.tile(tf.reshape(weights_symm, [1, input_dim, input_dim]),
                            [n_records, 1, 1],
                            name='weights_expand')

    # (Weight_expand .* Signal_expand) * (A^T)
    WS = tf.mul(weights_expand, signals_expand, name='mul_weight_signal')
    WSA = tf.batch_matmul(WS, tf.transpose(gA, [0, 2, 1]), name='mul_WS_gA')
    WSA_diag = tf.matrix_diag_part(WSA, name='diag_WSgA')

    # Biases definition and expansion
    biases = tf.Variable(tf.zeros([1, input_dim]),
                         name='biases')
    biases_expand = tf.tile(biases, [n_records, 1],
                            name='biases_expand')

    hidden = tf.nn.relu6(WSA_diag + biases_expand, name='hidden')
    hidden = tf.reshape(hidden, [n_records, 1, -1])

  return hidden

# signals_list = []
# for _ in np.arange(0, input_dim.eval()):
#   signals_list.append(signals)
# signals_expand = tf.concat(1, # concat dim = 1
#                           signals_list,
#                           name='signals_expand')
#
# # Weight expansion
# weights_list = []
# for _ in np.arange(0, n_records.eval()):
#   weights_list.append(weight_symm)
# weight_expand = tf.pack(weights_list,
#                         axis=2,
#                         name='weights_expand')
#
# biases_list = []
# for _ in np.arange(0, n_records.eval()):
#   biases_list.append(biases)
# biases_expand = tf.pack(biases_list,
#                         axis=0,
#                         name='biases_expand')

def softmax_linear_layer(signals, output_dim, name_scope):
  # Graph W - (batch_size, input_dim, input_dim)
  # Signals - (batch_size, 1, input_dim) --> (batch_size, input_dim)
  # Network W - (input_dim, input_dim) --> (batch_size, input_dim, input_dim)
  with tf.name_scope(name_scope):
    signals = tf.squeeze(signals, None, 'softmax_squeeze')
    input_dim = int(str(signals.get_shape()[1]))

    normal = tf.truncated_normal([input_dim, output_dim],
                          stddev=1.0 / np.sqrt(float(input_dim)))
    weights = tf.Variable(normal, name='weights')

    biases = tf.Variable(tf.zeros([output_dim]),
                         name='biases')
    logits = tf.matmul(signals, weights) + biases
    return logits


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('xentropy_mean', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.AdamOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1, name='CORRECT_IN_TOP_K')
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))