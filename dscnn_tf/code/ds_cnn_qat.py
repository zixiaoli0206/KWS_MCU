import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from util import dataset_refactor, tensor_eval, hex_to_c_array, tensor_eval_cnn, tf_weight_io_quant_converter
from joblib import load

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot

# Load dataset
def main(args):
    # Load dataset
    train_dict_data = load(args.traindict)
    val_dict_data = load(args.valdict)
    test_dict_data = load(args.testdict)
    train_feat_norm = load(args.trainfeat)
    val_feat_norm = load(args.valfeat)
    test_feat_norm = load(args.testfeat)
    train_dict_data['features'] = train_feat_norm
    val_dict_data['features'] = val_feat_norm
    test_dict_data['features'] = test_feat_norm

    train_feats, train_labels, _ = dataset_refactor(train_dict_data, 0)
    val_feats, val_labels, _ = dataset_refactor(val_dict_data, 1)
    test_feats, test_labels, _ = dataset_refactor(test_dict_data, 1)
    train_labels = train_labels[:, -1]
    val_labels = val_labels[:, -1]
    test_labels = test_labels[:, -1]

    # Evaluate tensorflow model performance
    tensor_path = 'models/kws12_dscnn_f32.h5'
    tensor_model = load_model(tensor_path)

    # Convert the model to a quantization aware model
    quant_aware_model = tfmot.quantization.keras.quantize_model(tensor_model)

    # `quantize_model` requires a recompile.
    quant_aware_model.compile(optimizer='adam',
                      loss=SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

    quant_aware_model.summary()

    # training parameters
    batchsize = 16
    epochs = 50
    early_stop_patience = 10

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        min_delta=0.001,  # Minimum change in the monitored quantity to qualify as an improvement
        patience=early_stop_patience,  # Number of epochs with no improvement after which training will be stopped
        mode='min',  # In 'min' mode, training will stop when the quantity monitored has stopped decreasing
        restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
    )

    history = quant_aware_model.fit(
        train_feats,
        train_labels,
        epochs=epochs,
        batch_size=batchsize,
        validation_data=(val_feats, val_labels),
        callbacks=[early_stopping]
    )

    test_feats = np.expand_dims(test_feats, axis=3)
    tf_weight_io_quant_converter(quant_aware_model, test_feats, 'dscnn_model_dyn_act_int8_tiny')