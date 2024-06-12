import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed
import torch
from util import dataset_refactor, gate_reorder, tf_full_precision_converter, tf_dyn_quant_converter, tf_weight_io_quant_converter
from joblib import load

def main(args):
    model_location = 'cpu'
    model_path = './models/SD_0_SNR_0_QF_0_FBW_12_LOG_1_NOR_standard_IN_16_L_2_H_48_D_0.20_CLA_GRU_FC_0_NC_12_QA_0_AQI_6_AQF_8_QW_0_WQI_1_WQF_7.pt'
    torch_model = torch.load(model_path, model_location)

    dict_torch_model = dict(torch_model.items())
    tensor_model = Sequential()
    tensor_model.add(GRU(units=48, return_sequences=True, input_shape=(61, 16)))
    tensor_model.add(GRU(units=48, return_sequences=True))
    tensor_model.add(TimeDistributed(Dense(12, activation='softmax')))

    for layer_idx, layer in enumerate(tensor_model.layers):
        if layer_idx == 0:
            weights_ih = dict_torch_model['cla.weight_ih_l0']
            weights_hh = dict_torch_model['cla.weight_hh_l0']
            bias_ih = dict_torch_model['cla.bias_ih_l0']
            bias_hh = dict_torch_model['cla.bias_hh_l0']
            bias_h = np.concatenate((np.expand_dims(bias_ih, axis=1), np.expand_dims(bias_hh, axis=1)), axis=1)
            weights_ih = weights_ih.T
            weights_hh = weights_hh.T
            bias_h = bias_h.T
            weights_ih, weights_hh, bias_h = gate_reorder(weights_ih, weights_hh, bias_h)
            layer.set_weights([weights_ih, weights_hh, bias_h])
        if layer_idx == 1:
            weights_ih = dict_torch_model['cla.weight_ih_l1']
            weights_hh = dict_torch_model['cla.weight_hh_l1']
            bias_ih = dict_torch_model['cla.bias_ih_l1']
            bias_hh = dict_torch_model['cla.bias_hh_l1']
            bias_h = np.concatenate((np.expand_dims(bias_ih, axis=1), np.expand_dims(bias_hh, axis=1)), axis=1)
            weights_ih = weights_ih.T
            weights_hh = weights_hh.T
            bias_h = bias_h.T
            weights_ih, weights_hh, bias_h = gate_reorder(weights_ih, weights_hh, bias_h)
            layer.set_weights([weights_ih, weights_hh, bias_h])
        if layer_idx == 2:
            fc_weight = dict_torch_model['fc_final.weight']
            fc_bias = dict_torch_model['fc_final.bias']
            fc_weight = fc_weight.T
            layer.set_weights([fc_weight, fc_bias])

    print('model load finished')

    test_dict_data = load(args.testdict)
    test_feat_norm = load(args.testfeat)
    test_dict_data['features'] = test_feat_norm

    # train_feats, train_labels, train_feat_len = dataset_refactor(train_dict_data)
    test_feats, test_labels, test_feat_len = dataset_refactor(test_dict_data, 1)

    test_dim_N = test_feats.shape[0]
    test_predict_prob = tensor_model.predict(test_feats)
    test_predict_seq = np.argmax(test_predict_prob, axis=2)
    test_predict = np.zeros(test_dim_N, dtype="float32")
    test_labels_squeeze = test_labels[:, 0]

    for i in range(0, test_dim_N):
        test_predict[i] = test_predict_seq[i, test_feat_len[i] - 1]

    # Compare the vectors element-wise
    comparison = test_predict == test_labels_squeeze
    # Count the number of True values
    same_elements_count = np.sum(comparison)
    # Calculate the accuracy
    accuracy = same_elements_count / len(test_predict)

    print("Accuracy:", accuracy)

    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    tensor_model.save('models/kws12_rnn_f32.h5')

    tf_weight_io_quant_converter(tensor_model, test_feats, 'rnn_model_dyn_act_int8_tiny')
    # tf_dyn_quant_converter(tensor_model)