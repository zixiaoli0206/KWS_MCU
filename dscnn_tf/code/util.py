import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import itertools

def idx_to_slice(lengths):
    """
    Get the index range of samples
    :param lengths: 1-D tensor containing lengths in time of each sample
    :return: A list of tuples containing the start & end indices of each sample
    """
    idx = []
    lengths_cum = np.cumsum(lengths)
    for i, len in enumerate(lengths):
        start_idx = lengths_cum[i] - lengths[i]
        end_idx = lengths_cum[i]
        idx.append((start_idx, end_idx))
    return idx


def slc(i, idx):
    return slice(idx[i][0], idx[i][1])


def dataset_refactor(dict_data, test_flag):
    # Split and form the dataset
    feature_slice_idx = idx_to_slice(dict_data['feature_lengths'])
    label_slice_idx = idx_to_slice(dict_data['target_lengths'])
    feat_stream = dict_data['features']
    target_stream = dict_data['targets']

    # Dimensions
    # N - Number of samples
    # T - Sequence length
    # F - Feature size
    dim_N = dict_data['feature_lengths'].shape[0]
    dim_F = dict_data['features'].shape[1]
    max_len = np.max(dict_data['feature_lengths'])

    # Batch shuffle
    if test_flag:
        s_idx = np.arange(dim_N)
    else:
        s_idx = np.random.permutation(dim_N)

    # Form the random training set
    feats_list = []
    labels_list = []
    for sample_idx in s_idx:
        feats_list.append(feat_stream[slc(sample_idx, feature_slice_idx), :])
        labels_list.append(target_stream[slc(sample_idx, label_slice_idx)])

    labels = np.zeros((dim_N, max_len), dtype='float32')
    feats = np.zeros((dim_N, max_len, dim_F), dtype='float32')
    feats_len = np.zeros(dim_N, dtype='int32')

    for i, sample in enumerate(feats_list):
        feats[i, :sample.shape[0], :] = sample
        labels[i, :sample.shape[0]] = labels_list[i].squeeze()
        feats_len[i] = sample.shape[0]

    return feats, labels, feats_len


def gate_reorder(weight_ih, weight_hh, bias):
    split_weight_ih = np.split(weight_ih, 3, axis=1)  # This splits the kernel into 3 equal parts along the second axis
    split_weight_hh = np.split(weight_hh, 3, axis=1)
    split_bias = np.split(bias, 3, axis=1)

    # Reordering from Reset-Update-New to Update-Reset-New for Keras
    # Current order indices: [Reset, Update, New] = [0, 1, 2]
    # Needed order indices for Keras: [Update, Reset, New] = [1, 0, 2]
    new_order = [1, 0, 2]
    new_weight_ih = np.concatenate([split_weight_ih[i] for i in new_order], axis=1)
    new_weight_hh = np.concatenate([split_weight_hh[i] for i in new_order], axis=1)
    new_bias = np.concatenate([split_bias[i] for i in new_order], axis=1)

    return new_weight_ih, new_weight_hh, new_bias


def tensor_eval(model, feats, labels, feat_len):
    test_dim_N = feats.shape[0]
    test_predict_prob = model.predict(feats)
    test_predict_seq = np.argmax(test_predict_prob, axis=2)
    test_predict = np.zeros(test_dim_N, dtype="float32")
    test_labels_squeeze = labels[:, 0]

    for i in range(0, test_dim_N):
        test_predict[i] = test_predict_seq[i, feat_len[i] - 1]

    # Compare the vectors element-wise
    comparison = test_predict == test_labels_squeeze
    # Count the number of True values
    same_elements_count = np.sum(comparison)
    # Calculate the accuracy
    accuracy = same_elements_count / len(test_predict)
    return accuracy, test_predict, test_labels_squeeze

def tensor_eval_cnn(model, feats, labels, feat_len):
    test_dim_N = feats.shape[0]
    test_predict_prob = model.predict(feats)
    test_predict = np.argmax(test_predict_prob, axis=1)
    test_labels_squeeze = labels[:, 0]


    # Compare the vectors element-wise
    comparison = test_predict == test_labels_squeeze
    # Count the number of True values
    same_elements_count = np.sum(comparison)
    # Calculate the accuracy
    accuracy = same_elements_count / len(test_predict)
    return accuracy, test_predict, test_labels_squeeze


def tf_full_precision_converter(model):
    # Convert the model to tensorflow without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
    ]
    converter._experimental_lower_tensor_list_ops = False  # Disable lowering tensor list ops.
    fp_tflite_model = converter.convert()

    # save model to disk
    open("models/kws12_model_f32.tflite", "wb").write(fp_tflite_model)


def tf_dyn_quant_converter(model):
    # Convert the model to tensorflow without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
    ]
    converter._experimental_lower_tensor_list_ops = False  # Disable lowering tensor list ops.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    fp_tflite_model = converter.convert()

    # save model to disk
    open("models/rnn_dynR.tflite", "wb").write(fp_tflite_model)

    c_model_name = 'rnn_dynR'
    # check if dir 'cfiles' exists, if not create it
    if not os.path.exists('cfiles'):
        os.makedirs('cfiles')
    # Write TFLite model to a C source (or header) file
    with open('cfiles/' + c_model_name + '.h', 'w') as file:
        file.write(hex_to_c_array(fp_tflite_model, c_model_name))

def tf_weight_io_quant_converter(model, dataset, filename):
    def representative_data_gen():
        for sample in dataset:
            # Model expects input shape to be [1, 61, 16], add batch dimension
            sample = np.expand_dims(sample, axis=0)
            yield [sample]

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,  # Only integer operations.
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS  # Allow TF ops to handle ops that can't be quantized.
    ]

    # Set the input and output tensors to uint8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert the model
    fp_tflite_model = converter.convert()

    # Save the model to disk
    tflite_name = "models/" + filename + ".tflite"
    with open(tflite_name, "wb") as f:
        f.write(fp_tflite_model)

    c_model_name = "cfiles/" + filename + ".h"
    # check if dir 'cfiles' exists, if not create it
    if not os.path.exists('cfiles'):
        os.makedirs('cfiles')
    # Write TFLite model to a C source (or header) file
    with open(c_model_name, 'w') as file:
        file.write(hex_to_c_array(fp_tflite_model, filename))
    print("model output")

# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):

    c_str = ''

    # Create header guard
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'

    # Add array length at top of file
    c_str += '\nstatic const unsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

    # Declare C variable
    c_str += 'static const unsigned char ' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data) :

        # Construct string from hex
        hex_str = format(val, '#04x')

        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '
        hex_array.append(hex_str)

    # Add closing brace
    c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

    # Close out header guard
    c_str += '#endif //' + var_name.upper() + '_H'

    return c_str

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = cm.T
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()