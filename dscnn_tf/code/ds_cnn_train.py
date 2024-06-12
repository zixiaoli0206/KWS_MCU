import os.path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from util import dataset_refactor
from joblib import load

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping

def main(args):
    # set global seeds for reproducibility
    tf.random.set_seed(1234)
    np.random.seed(1234)
    # Setting parameters for plotting
    plt.rcParams['figure.figsize'] = (15.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    print("TensorFlow version: ", tf.__version__)

    # check if GPU is available
    print("GPU is", "AVAILABLE" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

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

    # Define the DS-CNN model
    input_shape = (61, 16, 1)
    fp_model = tf.keras.models.Sequential([
        tf.keras.layers.ZeroPadding2D(padding=((5, 5), (1, 1)), input_shape=input_shape),
        tf.keras.layers.Conv2D(56, (10, 4), strides=(2, 2), use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
        tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(56, (1, 1), strides=(1, 1), use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
        tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(56, (1, 1), strides=(1, 1), use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
        tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(56, (1, 1), strides=(1, 1), use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
        tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(56, (1, 1), strides=(1, 1), use_bias=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.AveragePooling2D(pool_size=(25, 5), strides=(1, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(12, use_bias=True, activation='softmax')
    ])

    fp_model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    fp_model.summary()

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

    history = fp_model.fit(
        train_feats,
        train_labels,
        epochs=epochs,
        batch_size=batchsize,
        validation_data=(val_feats, val_labels),
        callbacks=[early_stopping]
    )

    # save the current model
    if not os.path.exists('models'):
        os.makedirs('models')
    fp_model.save('models/kws12_dscnn_f32.h5')

    fp_test_loss, fp_test_acc = fp_model.evaluate(test_feats, test_labels, verbose=2)

    converter = tf.lite.TFLiteConverter.from_keras_model(fp_model)
    fp_tflite_model = converter.convert()

    # save model to disk
    open("models/kws12_dscnn_f32.tflite", "wb").write(fp_tflite_model)

    print("DS-CNN training complete")

