# Soobeom Jang @ MCML, Yonsei University
# The code was modified from the tensorflow example.

"""Trains and Evaluates the Graph Shift network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import nn_model

from graph_signal_importer import Importer


## Basic model parameters as external flags.
#flags = tf.app.flags
#FLAGS = flags.FLAGS
#flags.DEFINE_float('learning_rate_gnn', 0.000001, 'Initial learning rate for gnn.')
#flags.DEFINE_float('learning_rate_nn', 0.000001, 'Initial learning rate for baseline nn.')
#flags.DEFINE_integer('max_steps', 50, 'Number of steps to run trainer.')
#flags.DEFINE_integer('batch_size', 10000, 'Batch size.  '
#                     'Must divide evenly into the dataset sizes.')
#flags.DEFINE_integer('summary_step', 1, 'Number of steps to print training summary. ')
#flags.DEFINE_integer('checkpoint_step', 5, 'Number of steps to evaluate model. ')
#flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

learning_rate_gnn = 0.0001
learning_rate_nn = 0.0001
max_steps = 50
batch_size = 10000
summary_step = 1
checkpoint_step = 5
train_dir = 'data/'

DATA_FILE = 'out_thresh_20161221134736.dat'  # mat-file storing data

def get_placeholders(input_dim, batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    input_dim: size of graph.
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    graphs_placeholder: graphs_placeholder.
    signals_placeholder: signals placeholder.
    labels_placeholder: Labels placeholder.
  """
  graphs_placeholder = tf.placeholder(tf.float32,
                                      shape=[batch_size, input_dim, input_dim],
                                      name='graphs_placeholder')
  signals_placeholder = tf.placeholder(tf.float32,
                                       shape=[batch_size, 1, input_dim],
                                       name='signals_placeholder')
  labels_placeholder = tf.placeholder(tf.int32,
                                      shape=[batch_size],
                                      name='labels_placeholder')
  return graphs_placeholder, signals_placeholder, labels_placeholder


def dataset_next_batch(importer, graphs_pl, signals_pl, labels_pl, batch_size):
  graphs_val, signals_val, labels_val = importer.next_batch(batch_size)
  feed_dict = {
    graphs_pl: graphs_val,
    signals_pl: signals_val,
    labels_pl: labels_val
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            importer,
            graphs_placeholder,
            signals_placeholder,
            labels_placeholder):
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  data_dimension = importer.get_dimension()
  steps_per_epoch = data_dimension[0] // batch_size
  num_examples = steps_per_epoch * batch_size
  for _ in range(steps_per_epoch):
    feed_dict = dataset_next_batch(importer,
                                  graphs_placeholder,
                                  signals_placeholder,
                                  labels_placeholder,
                                  batch_size)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def run_training():
  """Train Test dataset for a number of steps."""

  importer = Importer(DATA_FILE)
  graph_dim = importer.get_dimension()[1]

  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    graphs_placeholder, signals_placeholder, labels_placeholder = get_placeholders(
      graph_dim, batch_size)

    # Build a Graph that computes predictions from the inference model.
    # inference to evaluation

    # gnn-based model
    logits_gnn = nn_model.inference_gnn(graphs_placeholder, signals_placeholder, 2)
    loss_gnn = nn_model.loss(logits_gnn, labels_placeholder)
    train_op_gnn = nn_model.training(loss_gnn, learning_rate_gnn)  # learning operation
    eval_correct_gnn = nn_model.evaluation(logits_gnn, labels_placeholder)  # evaluation operation

    # ordinary nn based model
    logits_nn = nn_model.inference_nn(signals_placeholder, [50], 2)
    loss_nn = nn_model.loss(logits_nn, labels_placeholder)
    train_op_nn = nn_model.training(loss_nn, learning_rate_nn)
    eval_correct_nn = nn_model.evaluation(logits_nn, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    # summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()
    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in range(max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = dataset_next_batch(importer,
                                    graphs_placeholder,
                                    signals_placeholder,
                                    labels_placeholder,
                                    batch_size)
      feed_dict_nn = feed_dict.copy()
      feed_dict_nn.pop(graphs_placeholder, None)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value_gnn = sess.run([train_op_gnn, loss_gnn],
                                   feed_dict=feed_dict)
      _, loss_value_nn = sess.run([train_op_nn, loss_nn],
                               feed_dict=feed_dict_nn)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % summary_step == 0:
        # Print status to stdout.
        print('GNN: Step %d: loss = %.2f (%.3f sec)' % (step, loss_value_gnn, duration))
        print('NN : Step %d: loss = %.2f (%.3f sec)' % (step, loss_value_nn, duration))
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % checkpoint_step == 0 or (step + 1) == max_steps:
        save_path = saver.save(sess, train_dir, global_step=step)
        print('Model saved in file: %s' % save_path)

        # Evaluate against the training set.
        print('Validation Data Eval:')
        print('GNN')
        do_eval(sess,
                eval_correct_gnn,
                importer,
                graphs_placeholder,
                signals_placeholder,
                labels_placeholder)
        print('NN')
        do_eval(sess,
                eval_correct_nn,
                importer,
                graphs_placeholder,
                signals_placeholder,
                labels_placeholder)


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
