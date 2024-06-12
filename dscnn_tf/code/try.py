import numpy as np
import tensorflow as tf
import os
import pandas as pd

def hex_to_c_array(hex_data, var_name):
    c_str = ""
    hex_array = ["0x{:02x}".format(byte) for byte in hex_data]
    for i in range(0, len(hex_array), 12):
        c_str += "    " + ", ".join(hex_array[i:i+12]) + ",\n"
    c_str = c_str.rstrip(",\n")
    c_str = (
        "const unsigned char {}[] = {{\n"
        "{}\n"
        "}};\n".format(var_name, c_str)
    )
    return c_str

def tf_weight_io_quant_converter(model, dataset):
    def representative_data_gen():
        for sample in dataset:
            # Ensure sample shape matches model input shape: (64, 16, 1)
            sample = sample.astype(np.float32)  # Ensure the sample is of type FLOAT32
            sample = np.expand_dims(sample, axis=-1)  # Add channel dimension if necessary
            sample = np.expand_dims(sample, axis=0)  # Add batch dimension
            yield [sample]

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [
        # tf.lite.OpsSet.TFLITE_BUILTINS_INT8,  # Only integer operations.
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS  # Allow TF ops to handle ops that can't be quantized.
    ]

    # Set the input and output tensors to int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert the model
    fp_tflite_model = converter.convert()

    # Save the model to disk
    with open("models/kws12_model_int8_2.tflite", "wb") as f:
        f.write(fp_tflite_model)

    c_model_name = 'dscnn_model_dyn_act_int8_neuro56'
    # Check if dir 'cfiles' exists, if not create it
    if not os.path.exists('cfiles'):
        os.makedirs('cfiles')
    # Write TFLite model to a C source (or header) file
    with open('cfiles/' + c_model_name + '.h', 'w') as file:
        file.write(hex_to_c_array(fp_tflite_model, c_model_name))

# Example model and dataset
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
    tf.keras.layers.Dense(12, use_bias=True)
])

# Assuming test_feats is your dataset and is of shape (n_samples, 64, 16)
test_feats = np.random.rand(100, 61, 16).astype(np.float32)  # Example dataset, ensure type FLOAT32

# Convert the model
tf_weight_io_quant_converter(fp_model, test_feats)
