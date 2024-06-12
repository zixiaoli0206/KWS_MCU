import os
import tensorflow as tf
# Show the model size for the non-quantized HDF5 model
fp_h5_in_kb = os.path.getsize('models/kws12_model_f32_transfer.h5') / 1024
print("HDF5 Model size without quantization: %d KB" % fp_h5_in_kb)

# Show the model size for the non-quantized TFLite model
fp_tflite_in_kb = os.path.getsize('models/kws12_model_f32.tflite') / 1024
print("TFLite Model size without quantization: %d KB" % fp_tflite_in_kb)

# Show the model size for the weight quantized TFLite model
fp_tflite_dynR = os.path.getsize('models/kws12_model_dynR.tflite') / 1024
print("TFLite Model size with weight quantization: %d KB" % fp_tflite_dynR)

# Determine the reduction in model size
# print("\nReduction in file size by a factor of %f" % (fp_h5_in_kb / fp_tflite_in_kb))