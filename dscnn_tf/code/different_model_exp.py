import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed
import torch
from util import dataset_refactor, gate_reorder, tf_full_precision_converter, tf_dyn_quant_converter, tf_weight_io_quant_converter
from joblib import load

# fp_model = tf.keras.models.Sequential([
#     # tf.keras.layers.Input(shape=(61, 16), name='input'),
#     tf.keras.layers.LSTM(48, input_shape=(61, 16), unroll=False)
# ])
# train_dict_data = load('../dataset/train_dict_data.joblib')
test_dict_data = load('../dataset/test_dict_data.joblib')
test_feat= load('../dataset/test_feat.joblib')
test_dict_data['features'] = test_feat

# train_feats, train_labels, train_feat_len = dataset_refactor(train_dict_data)
test_feats, test_labels, test_feat_len = dataset_refactor(test_dict_data, 1)
test_feats = np.expand_dims(test_feats, axis=3)

fp_model = tf.keras.models.Sequential([
    tf.keras.layers.ZeroPadding2D(padding=((5, 5), (1, 1)), input_shape=(64, 16, 1)),
    tf.keras.layers.Conv2D(64, (10, 4), strides=(2, 2), use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
    tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
    tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
    tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
    tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.AveragePooling2D(pool_size=(25, 5), strides=(1, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(12, use_bias=True)
])

# tf_dyn_quant_converter(fp_model)
tf_weight_io_quant_converter(fp_model, test_feats)