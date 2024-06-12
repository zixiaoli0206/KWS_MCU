import tensorflow as tf

# Define a simple model with a GRU layer
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, 10)),  # Example input shape: (batch, sequence, features)
    tf.keras.layers.GRU(32),  # GRU layer with 32 units
    tf.keras.layers.Dense(1)  # Output layer
])

# Assume model is compiled and trained here

# Setup the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # Enable TensorFlow ops.
]
converter._experimental_lower_tensor_list_ops = False  # Disable lowering tensor list ops.

# Convert the model
tflite_model = converter.convert()

# Save the TensorFlow Lite model to file
with open('my_gru_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Success")