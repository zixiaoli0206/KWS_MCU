import matplotlib.pyplot as plt
import numpy as np
import serial
import sys
import time
from matplotlib import style

# Applying a style to the plots
style.use('ggplot')  # You can choose other styles like 'seaborn-darkgrid', 'ggplot', etc.

def plot_results(index, true_label, predicted_label, status_text, accuracy):
    ax.clear()  # Clear previous results
    # Display results or status text based on the flag
    if true_label and predicted_label:
        ax.text(0.5, 0.6, f"True Label: {true_label}\nPredicted Label: {predicted_label}",
                horizontalalignment='center', verticalalignment='center',
                fontsize=14, color='navy', transform=ax.transAxes)
    ax.text(0.5, 0.1, status_text,
            horizontalalignment='center', verticalalignment='center',
            fontsize=12, color='darkred', transform=ax.transAxes)
    # Display accuracy
    ax.text(0.5, 0.3, f"Accuracy: {accuracy:.2%}",
            horizontalalignment='center', verticalalignment='center',
            fontsize=12, color='green', transform=ax.transAxes)
    ax.axis('off')  # Hide axes
    plt.draw()

def plot_processing():
    ax.clear()  # Clear previous results
    ax.text(0.5, 0.1, 'MCU is busy',
            horizontalalignment='center', verticalalignment='center',
            fontsize=12, color='darkred', transform=ax.transAxes)
    ax.axis('off')  # Hide axes
    plt.draw()
    plt.pause(0.1)

def send_next_sample(i, x_test, y_test, ser, classes, button, correct_predictions, total_predictions):
    plot_processing()
    button.set_active(False)  # Disable the button while processing
    x = x_test[i[0]].astype(np.int8)
    true_class_idx = y_test[i[0]][0]
    true_label = classes[true_class_idx]
    ser.write(x.tobytes())
    ser.flush()
    pred = np.array([])
    while np.size(pred) == 0:
        time.sleep(1)
        if ser.inWaiting() > 0:
            pred_buf = ser.read(ser.inWaiting())
            if len(pred_buf) >= 12:
                pred = np.frombuffer(pred_buf[-12:], dtype=np.int8)
    predicted_label = classes[np.argmax(pred)]
    if true_label == predicted_label:
        correct_predictions[0] += 1
    total_predictions[0] += 1
    accuracy = correct_predictions[0] / total_predictions[0]
    plot_results(i[0], true_label, predicted_label, 'MCU is ready', accuracy)
    i[0] += 1
    button.set_active(True)

if __name__ == '__main__':
    x_test = np.load('../dscnn_tf/code/data/feat_kws.npy')
    y_test = np.load('../dscnn_tf/code/data/label_kws.npy').squeeze()
    np.random.seed(1)
    perm = np.random.permutation(len(y_test))
    x_test, y_test = x_test[perm], y_test[perm]
    classes = ['silence', 'unknown', 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    ser = serial.Serial(port='/dev/tty.usbmodem1103', baudrate=115200, timeout=3)
    ser.flushInput()
    fig, ax = plt.subplots()
    ax.axis('off')
    plt.subplots_adjust(bottom=0.3)
    ax.set_position([0.1, 0.2, 0.8, 0.7])  # Adjust plot area
    current_sample_index = [0]
    correct_predictions = [0]
    total_predictions = [0]
    button_ax = plt.axes([0.4, 0.05, 0.2, 0.1])  # Button positioning
    bnext = plt.Button(button_ax, 'Next Sample', color='lightblue', hovercolor='0.975')
    bnext.on_clicked(lambda event: send_next_sample(current_sample_index, x_test, y_test, ser, classes, bnext, correct_predictions, total_predictions))
    plt.show()
