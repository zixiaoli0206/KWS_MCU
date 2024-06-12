__author__ = "Zixiao Li"
__copyright__ = "Copyright 2024"
__credits__ = ["Chang Ga"]
__license__ = "Private"
__version__ = "0.1.0"
__maintainer__ = "Zixiao Li"
__email__ = "zixili@ethz.ch"
__status__ = "Prototype"

import os
import argparse
import importlib
import sys

import ds_cnn_train
import ds_cnn_qat
import eval_dscnn
import rnn_transfer
import eval_rnn
import model_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', default='ds_cnn_train', help='Which step to start from')
    path_root = os.path.dirname(os.path.abspath(__file__))
    args = parser.parse_args()
    # The dataset is generated from another uploaded repo from GSC v2
    args.traindict = '../dataset/train_dict_data.joblib'
    args.valdict = '../dataset/val_dict_data.joblib'
    args.testdict = '../dataset/test_dict_data.joblib'
    args.trainfeat = '../dataset/train_feat.joblib'
    args.valfeat = '../dataset/val_feat.joblib'
    args.testfeat = '../dataset/test_feat.joblib'

    # Train ds-cnn model and store the fp32 version
    if args.step == 'ds_cnn_train':
        print("####################################################################################################")
        print("# Step 0: Training DS_CNN                                                                          #")
        print("####################################################################################################")
        ds_cnn_train.main(args)

    # Quantization-aware training for ds-cnn and store int8 version
    if args.step == 'ds_cnn_qat':
        print("####################################################################################################")
        print("# Step 1: Quantization-aware training for DS_CNN                                                   #")
        print("####################################################################################################")
        ds_cnn_qat.main(args)

    # Evaluate the ds-cnn model and save the quantized dataset
    if args.step == 'eval_ds_cnn':
        print("####################################################################################################")
        print("# Step 2: Evaluate DS_CNN                                                                          #")
        print("####################################################################################################")
        eval_dscnn.main(args)

    # Transfer the trained RNN-LSTM/GRU model to tensorflow and save the fp32 version
    if args.step == 'rnn_transfer':
        print("####################################################################################################")
        print("# Step 3: Transfer the trained pytorch rnn model to tensorflow                                     #")
        print("####################################################################################################")
        rnn_transfer.main(args)

    # Evaluate the RNN model and save the int8 version.
    if args.step == 'eval_rnn':
        print("####################################################################################################")
        print("# Step 4: Evaluate transferred RNN                                                                 #")
        print("####################################################################################################")
        eval_rnn.main(args)

    # Evaluate the model size and count the MACs
    if args.step == 'model_size':
        print("####################################################################################################")
        print("# Step 5: Model size and MAC number                                                                #")
        print("####################################################################################################")
        # test dscnn with dscnn=1; test rnn with dscnn = 0
        model_size.main(dscnn=1)