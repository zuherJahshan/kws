# %%
from matplotlib import pyplot as plt
import IPython.display as ipd
import tensorflow as tf
from tensorflow import keras

# %%
from dataset import get_datasets
from its_safoos import ITS

# %%
#### hyper parameters that defines the structure of the model
num_classes = 31 # ds.get_labels()
sampled_frequencies = 129 # the number of frequency samples

learning_rate = 0.001
weight_decay = 0.005
batch_size = 64
epochs = 60
# patch_size = 6  # Size of the patches to be extract from the input images
# num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 1
mlp_head_units = [
    526,
    256,
]  # Size of the dense layers of the final classifier


# %%
train, valid, test = get_datasets(batch_size=batch_size, type='mfccs')

# %%
# Run both models TCResNet and StateTransformer for 30 epochs and graph the accuracy results
import matplotlib.pyplot as plt
import csv

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
        print("Altering data at index", data_idx, "with value", logs['val_accuracy'])
        data[data_idx[0]][data_idx[1]] = str(logs['val_accuracy'])
        with open(self.filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(model_names)
            writer.writerows(data)


import csv


def init_results_file(filename, repeats_to_examine, state_cells_to_examine, epochs):
    model_names = []
    for num_repeats, num_state_cells in zip(repeats_to_examine, state_cells_to_examine):    
        model_names.append(f"r={num_repeats},s={num_state_cells}")
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(model_names)
        writer.writerows([[0] * len(model_names)] * epochs)


results_filename = 'results_safoos.csv' 
repeats_to_examine =        [1, 2, 2]
state_cells_to_examine =    [1, 4, 10]
init_results_file(results_filename, repeats_to_examine, state_cells_to_examine, epochs)
for num_repeats, num_state_cells in zip(repeats_to_examine, state_cells_to_examine):
    state_transformer = ITS(
        num_classes=31,
        num_repeats=num_repeats,
        num_heads=8,
        num_state_cells=num_state_cells,
        input_seq_size=31,
        projection_dim=32,
        inner_ff_dim=64,
        dropout=0.1,
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
    )

    state_transformer.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


    model_path = "./models/its_chkpnt/its_chkpnt.ckpt"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
    )

    state_transformer_history = state_transformer.fit(
        train,
        validation_data=valid,
        epochs=epochs,
        callbacks=[
            ResultsWriter(results_filename, num_repeats, num_state_cells),
        ],
    )


# %%



