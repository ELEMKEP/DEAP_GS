import os
import sys
import getopt

import pickle
import datetime
import numpy as np


def get_current_datetime_str():
  dt = datetime.datetime.now()
  return '%d%02d%02d%02d%02d%02d' % (
    dt.year,
    dt.month,
    dt.day,
    dt.hour,
    dt.minute,
    dt.second)


def main(argv):
  input_file = os.path.abspath('out_20161221132234.dat')
  output_file = 'out_thresh_' + get_current_datetime_str() + '.dat'
  graph_thresh_type = 'percentile'
  graph_thresh_val = 50

  # In progress.
  try:
    opts, args = getopt.getopt(argv,
                               'hi:o:t:v:',
                               ['input=', 'output=', 'thresh_type=', 'thresh_val='])
  except getopt.GetoptError:
    print('draft_thresholding.py -g <connectivity_method> -s <signal_method>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print('draft_thresholding.py -i <input_file> -o <output_file>',
      '-t <thresh_type> -v <thresh_val>')
      sys.exit()
    elif opt in ('-i', '--input'):
      input_file = os.path.abspath(arg)
    elif opt in ('-o', '--output'):
      output_file = arg
    elif opt in ('-t', '--thresh_type'):
      graph_thresh_type = arg
    elif opt in ('-v', '--thresh_val'):
      graph_thresh_val = float(arg)
  print('Selected threshold type: %s' % graph_thresh_type)
  print('Selected threshold value: %.2f' % graph_thresh_val)

  print('file_name: ' + input_file)
  print('Loading dataset...')
  with open(input_file, 'rb') as f:
    data = pickle.load(f, encoding='bytes')
  print('Dataset loading complete.')

  # Graph thresholding  
  graphs = data['rs_graphs']
  signals = data['rs_signals']
  labels = data['rs_labels']

  print('Graph thresholding process started.')
  for idx in range(graphs.shape[0]):
    if idx % 100 == 0 and idx != 0:
      print('%d graphs processed' % (idx+1))

    graph_values = graphs[idx].reshape(-1)
    if graph_thresh_type == 'percentile':
      g_val_thresh = np.percentile(graph_values, graph_thresh_val)
    elif graph_thresh_type == 'absolute':
      max_val = np.max(np.sort(graph_values))
      min_val = np.min(np.sort(graph_values))
      assert (min_val < graph_thresh_val & max_val > graph_thresh_val), 'Outliered threshold value!'
      g_val_thresh = graph_thresh_val
    else:
      assert False, 'Not supported thresholding type.'

    graph_values[graph_values < g_val_thresh] = 0
    graph_values[graph_values >= g_val_thresh] = 1
  print('Graph thresholding complete.')

  # Label thresholding
  print('Label thresholding process started.')
  label_thresh = np.percentile(labels, 50)
  labels[labels < label_thresh] = 0
  labels[labels >= label_thresh] = 1
  print('Label thresholding process completed.')

  rs_dict = dict()
  rs_dict['rs_graphs'] = graphs
  rs_dict['rs_signals'] = signals
  rs_dict['rs_labels'] = labels

  print('Saving thresholded dataset...')
  with open(output_file, 'wb') as f:
    pickle.dump(rs_dict, f)
  print('Complete.')

if __name__ == "__main__":
    main(sys.argv)