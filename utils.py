import tensorflow as tf
from tensorflow import keras
import csv
import os


def init_results_file(filename, repeats_to_examine, state_cells_to_examine, epochs):
    model_names = []
    for num_repeats, num_state_cells in zip(repeats_to_examine, state_cells_to_examine):    
        model_names.append(f"r={num_repeats},s={num_state_cells}")
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(model_names)
        writer.writerows([[0] * len(model_names)] * epochs)

class ResultsWriter(keras.callbacks.Callback):
    def __init__(self, filename, repeats, state_cells):
        self.filename = filename
        self.repeats = repeats
        self.state_cells = state_cells

    
    def get_file_lines(self):
        model_names = []
        data = []
        
        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            first_row = True
            for row in reader:
                if first_row:
                    model_names = row
                    first_row = False
                else:
                    data.append(row)

        return model_names, data


    def get_data_idx_to_alter(self, model_names, epochs):
        return epochs, model_names.index(f"r={self.repeats},s={self.state_cells}"), 
    
    
    def on_epoch_end(self, epoch, logs=None):
        model_names, data = self.get_file_lines()
        data_idx = self.get_data_idx_to_alter(model_names, epoch)
        data[data_idx[0]][data_idx[1]] = str(logs['val_accuracy'])
        with open(self.filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(model_names)
            writer.writerows(data)



def schedule(epoch, lr):
    drop_rate = 0.8
    epochs_drop = 10
    if epoch % epochs_drop == 0:
        return lr * drop_rate
    return lr


lrs = tf.keras.callbacks.LearningRateScheduler(
    schedule,
    verbose=0
)