# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:25:12 2016

Draft to implement graph signalization of DEAP dataset.

@author: Soobeom Jang, Yonsei University
"""
import os
import xlrd
import numpy as np
import cPickle
import sys
import getopt
import datetime
import gc

nTrial = 40 # specified in DEAP description
nSubject = 32 # specified in DEAP description
channelDim = 32 # Selected
nTimestamp = 20 # Selected parameter
sRate = 128 # specified in DEAP description
baseline = 3 # in seconds. Specified in DEAP description
nWindow = sRate * baseline

def get_current_datetime_str():
    dt = datetime.datetime.now()
    return '%d%02d%02d%02d%02d%02d' % (
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            dt.second)

def conn_xcorr(data):
    """Calculate the connectivity to construct graph adjacency matrix by cross-correlation.

    Keyword arguments:
        data -- Original data

    Return arguments:
        G -- Calculated connectivity

    """

    nTrial = data.shape[0]
    channelDim = data.shape[1]
    G = np.zeros([nTrial, channelDim, channelDim])

    for trial_idx in range(nTrial):
        trial_data = data[trial_idx]
        for (idx1, idx2) in zip(range(channelDim), range(channelDim)):
            xcorr = np.correlate(trial_data[idx1], trial_data[idx2], 'full')
            G[trial_idx][idx1][idx2] = np.max(xcorr)

    G = (G + np.transpose(G, (0, 2, 1)))/2 # to guarantee symmetricity
    return G

def sig_timestamp_power(data):
    """Calculate the graph signal to apply to the graph by timestamp power.

    Keyword arguments:
        data -- Original data

    Return arguments:
        signal -- Calculated signal
    """
    # function template for signal power
    return np.mean(np.square(data), axis=2)

# openning participant excel file and storing them to dict
# keys[0:4]: participant_id, Trial, Experiment_id, Start_time,
# keys[4:9]: Valence, Arousal, Dominance, Liking, Familiarity

def main(argv):
    graph_method = 'xcorr'
    signal_method = 'power'
    output_file = 'out_'+get_current_datetime_str()+'.dat'
    DEAP_path = 'E:\DEAP' # For my computer.

    if gc.isenabled():
        gc.enable()

    # In progress.
    try:
        opts, args = getopt.getopt(argv,
                                   'hd:o:g:s:',
                                   ['directory=', 'output=', 'graph=', 'signal='])
    except getopt.GetoptError:
        print 'draft.py -g <connectivity_method> -s <signal_method>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'draft.py -d <dataset_directory> -o <output_prefix>',
            '-g <connectivity_method> -s <signal_method>'
            sys.exit()
        elif opt in ('-d', '--directory'):
            DEAP_path = os.path.abspath(arg)
        elif opt in ('-o', '--output'):
            output_file = arg
        elif opt in ('-g', '--graph'):
            graph_method = arg
        elif opt in ('-s', '--signal'):
            signal_method = arg
    print 'Selected connectivity method: %s' % graph_method
    print 'Selected signal method: %s' % signal_method

    # Opening DEAP directory
    try:
        wb = xlrd.open_workbook(os.path.join(DEAP_path,
                                             'participant_ratings.xls'))
        rating_data = {}
        worksheet = wb.sheet_by_index(0)
        for idx in range(worksheet.ncols):
            rating_data[worksheet.cell(0, idx)] = worksheet.col(idx, 1)
        wb.release_resources()
    except IOError:
        print 'DEAP rating file not found!'
        sys.exit(2)

    # Result data variables
    rs_graphs = np.zeros((0, channelDim, channelDim))
    rs_signals = np.zeros((0, channelDim))
    rs_labels = np.zeros((0, 1))

    # Signal processing per subjects
    for idx in range(nSubject):
        print 'Current subject No. %d' % (idx+1)
        # pickle load data
        try:
            data_path = os.path.join(DEAP_path, 's%02d.dat' % (idx+1))
            data_file = cPickle.load(open(data_path, 'rb'))
        except IOError:
            print 'Current DEAP data files not found!'
            sys.exit(2)

        data = data_file['data']
        labels = data_file['labels']

        # baseline signal extraction
        data_baseline = data[np.ix_(range(data.shape[0]),
                                    range(channelDim),
                                    range(sRate * baseline))]
        bSignal = np.mean(np.square(data_baseline), axis=2)
        bSignal_ext = np.repeat(bSignal, nTimestamp, axis=0)
        # shape: nTrial, channelDim

        # making data slice which is actually used
        data_slice = data[np.ix_(range(data.shape[0]),
                                 range(channelDim),
                                 range(sRate * baseline, data.shape[2]))]
        data_slice = data_slice.reshape(nTrial,
                                        channelDim,
                                        nTimestamp,
                                        nWindow)
        data_slice = data_slice.transpose((0, 2, 1, 3))
        data_slice = data_slice.reshape(nTrial*nTimestamp,
                                        channelDim,
                                        nWindow)

        # making topology and graph signal
        G = conn_xcorr(data_slice)
        # shape: nTrial x nTimestamp, channelDim, channelDim
        gSignal = sig_timestamp_power(data_slice)
        # shape: nTrial x nTimestamp, channelDim
        gSignal = gSignal - bSignal_ext

        # get label
        labels_ext = np.repeat(labels, nTimestamp, axis=0)
        labels_ext = np.reshape(labels_ext[:, 0], (-1, 1))

        rs_graphs = np.concatenate((rs_graphs, G), axis=0)
        rs_signals = np.concatenate((rs_signals, gSignal), axis=0)
        rs_labels = np.concatenate((rs_labels, labels_ext), axis=0)

    # Storing result data in dictionary to save in pickle
    rs_dict = {}
    rs_dict['rs_graphs'] = rs_graphs
    rs_dict['rs_signals'] = rs_signals
    rs_dict['rs_labels'] = rs_labels

    with open(output_file, 'wb') as f:
        cPickle.dump(rs_dict, f)

if __name__ == "__main__":
    main(sys.argv)






