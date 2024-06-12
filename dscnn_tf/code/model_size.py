import tensorflow as tf
from tensorflow.keras.models import load_model
# Calculate MACs function
def calc_macs(layer, input_shape):
    output_shape = layer.output_shape
    if isinstance(layer, tf.keras.layers.Conv2D):
        # For regular convolution
        kernel_height, kernel_width = layer.kernel_size
        in_channels = input_shape[-1]
        out_channels = layer.filters
        return output_shape[1] * output_shape[2] * kernel_height * kernel_width * in_channels * out_channels
    elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        # For depthwise convolution
        kernel_height, kernel_width = layer.kernel_size
        in_channels = input_shape[-1]
        return output_shape[1] * output_shape[2] * kernel_height * kernel_width * in_channels
    elif isinstance(layer, tf.keras.layers.Dense):
        # For dense layers
        input_units = layer.input_shape[-1]
        output_units = layer.units
        return input_units * output_units
    elif isinstance(layer, tf.keras.layers.SimpleRNN):
        # For RNN layers
        input_units = input_shape[-1]
        output_units = layer.units
        # MACs for the input weights and recurrent weights
        macs_input_weights = input_units * output_units
        macs_recurrent_weights = output_units * output_units
        return (macs_input_weights + macs_recurrent_weights) * output_shape[1]
    elif isinstance(layer, tf.keras.layers.GRU):
        # For GRU layers (3 gates: update, reset, and new gates)
        input_units = input_shape[-1]
        output_units = layer.units
        # Each gate has input weights and recurrent weights
        macs_per_gate = input_units * output_units + output_units * output_units
        total_macs = 3 * macs_per_gate
        return total_macs * output_shape[1]
    elif isinstance(layer, tf.keras.layers.LSTM):
        # For LSTM layers (4 gates: forget, input, cell state, output)
        input_units = input_shape[-1]
        output_units = layer.units
        # Each gate has input weights and recurrent weights
        macs_per_gate = input_units * output_units + output_units * output_units
        total_macs = 4 * macs_per_gate
        return total_macs * output_shape[1]
    return 0

def main(dscnn):
    if dscnn:
        tensor_path = 'models/kws12_dscnn_f32.h5'
    else:
        tensor_path = 'models/kws12_rnn_f32.h5'
    model = load_model(tensor_path)

    # Calculate total MACs for the model
    total_macs = 0
    input_shape = model.input_shape[1:]  # Initial input shape

    for layer in model.layers:
        macs = calc_macs(layer, input_shape)
        input_shape = layer.output_shape   # Update input shape to output shape of current layer
        total_macs += macs

    # Output total parameters and MACs
    print(f"Total parameters in the model: {model.count_params()}")
    print(f"Total MACs in the model: {total_macs}")

