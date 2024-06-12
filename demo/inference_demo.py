import matplotlib.pyplot as plt
import numpy as np
import serial
import sys
import time

if __name__ == '__main__':
    x_test = np.load('../dscnn_tf/code/data/feat_kws.npy')
    y_test = np.load('../dscnn_tf/code/data/label_kws.npy').squeeze()

    # Randomly permute the test set
    np.random.seed(1)
    perm = np.random.permutation(len(y_test))
    x_test = x_test[perm]
    y_test = y_test[perm]

    # dataset_name = sys.argv[1]
    classes = ['silence', 'unknown', 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

    print(f'Loaded x with shape: {x_test.shape}')
    print(f'Loaded y with shape: {y_test.shape}')

    ser = serial.Serial(port='/dev/tty.usbmodem1103', baudrate=115200, timeout=3)
    # flush the serial port
    ser.flush()
    ser.flushInput()
    ser.flushOutput()

    # Test Accuracy on MCU
    correct_count = 0
    # define how many images from the test set to send to the MCU
    test_len = 20
    # get how many prediction we iterated over
    num_pred = 0

    for x, y in zip(x_test[:test_len], y_test[:test_len]):
        pred = []
        x = x.astype(np.int8)
        num_pred += 1
        class_idx = y[0]
        print("True class: \n {}".format(classes[class_idx]))
        ser.write(x.tobytes())
        while np.size(pred) == 0:
            time.sleep(1)
            pred_buf = ser.read(12)
            pred = np.frombuffer(pred_buf, dtype=np.int8)
        if (np.argmax(pred) == class_idx):
            correct_count += 1
        print("Predicted class: \n {}".format(classes[np.argmax(pred)]))

    print(correct_count)