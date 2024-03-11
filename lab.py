from matplotlib import pyplot as plt
import IPython.display as ipd
import tensorflow as tf
from tensorflow import keras
import csv

from dataset import get_datasets, labels_v1, labels_v2
from its_lru import StateTransformer


version = 2
num_classes = len(labels_v1) if version == 1 else len(labels_v2)
sampled_frequencies = 129 # the number of frequency samples

learning_rate = 0.001 / 5
weight_decay = 0.005
batch_size = 64
num_epochs = 10000  # For real training, use num_epochs=100. 10 is a test value
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

train, valid, test = get_datasets(batch_size=batch_size, type='mfccs', version=version)

# %%
# Run both models TCResNet and StateTransformer for 30 epochs and graph the accuracy results
results = {}
for num_state_cells in [1, 4, 8, 12]:
    print("++++++++++++++++++++++++++++")
    print("Started training for num_state_cells: ", num_state_cells)
    state_transformer = StateTransformer(
        num_classes=num_classes,
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
        epochs=21,
        # callbacks=[
        #     model_checkpoint_callback,
        # ],
    )
    results[num_state_cells] = state_transformer_history.history['val_accuracy']

# write results to a csv file
with open('results.csv', 'w') as f:
    csv_writer = csv.writer(f)
    for key, values in results.items():
        csv_writer.writerow([key] + values)