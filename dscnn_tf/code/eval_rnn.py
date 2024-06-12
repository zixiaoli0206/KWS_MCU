import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from util import dataset_refactor, tensor_eval, hex_to_c_array, tensor_eval_cnn
from joblib import load
from util import gate_reorder, tf_full_precision_converter, tf_dyn_quant_converter, tf_weight_io_quant_converter, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import pandas as pd
from tensorflow.keras.models import load_model

# Load data
def main(args):
    test_dict_data = load(args.testdict)
    test_feat_norm = load(args.testfeat)
    test_dict_data['features'] = test_feat_norm

    test_feats, test_labels, test_feat_len = dataset_refactor(test_dict_data, 1)

    # Evaluate tensorflow model performance
    tensor_path = 'models/kws12_rnn_f32.h5'
    tensor_model = load_model(tensor_path)
    tensor_accuracy, tf_predict, tf_label = tensor_eval(tensor_model, test_feats, test_labels, test_feat_len)

    # Load the TFLite model and allocate tensors.
    tflite_path = 'models/rnn_model_dyn_act_int8_tiny.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input and output tensor details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assuming test_feats is your test data with shape [512, 60, 16]
    # Prepare to store predictions
    num_samples = test_feats.shape[0]
    num_frames = test_feats.shape[1]
    lite_pred = np.zeros((num_samples, num_frames), dtype=int)

    # Iterate over each sample
    for sample_idx in range(num_samples):
        # Initialize RNN states if necessary (not applicable here since states are handled internally)
        lite_feat = test_feats[sample_idx, :, :]
        lite_feat = np.expand_dims(lite_feat, axis=0)

        if input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = input_details[0]["quantization"]
            lite_feat = lite_feat / input_scale + input_zero_point

        lite_feat = lite_feat.astype(input_details[0]['dtype'])
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], lite_feat)

        # Run the model
        interpreter.invoke()

        # Get the output prediction
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        lite_pred[sample_idx, :] = np.argmax(output, axis=1)

    # Now tf_pred should contain the correct predictions for each frame in each sample
    lite_pred_final = lite_pred[:, -1]
    test_label_final = test_labels[:, -1]
    lite_pred_comparison = test_label_final == lite_pred_final
    lite_same_elements_count = np.sum(lite_pred_comparison)
    lite_accuracy = lite_same_elements_count / len(lite_pred_comparison)

    print("tensorflow accuracy:", tensor_accuracy)
    print("tensorflow lite accuracy:", lite_accuracy)

    ## Confusion matrix and plot
    cm = confusion_matrix(test_label_final, lite_pred_final, labels=[i for i in range(0, 12)])
    # cm = confusion_matrix(tf_label, tf_predict, labels=[i for i in range(0, 12)])
    classes = ['_silence_', '_unknown_', 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    series_with_index = pd.Series(classes)
    plot_confusion_matrix(cm, series_with_index, True, 'RNN Weight INT8', cmap=plt.cm.Spectral.reversed())
