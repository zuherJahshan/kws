import tensorflow_datasets as tfds
import tensorflow as tf

labels_v1 = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
    "_silence_"
]

labels_v2 = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
    "_silence_"
]

FREQUENCY = 16_000
DURATION = FREQUENCY # Which means 2 seconds

def get_audio_and_label(x, version):
    audio = x["audio"][:DURATION]
    label = x["label"]
    if tf.shape(audio)[0] < DURATION:
        audio = tf.pad(audio, paddings=[[DURATION - tf.shape(audio)[0], 0]])
    
    # the label is an int inside [0, 30]. Please convert it to a one-hot vector of size 31
    label = tf.one_hot(label, len(labels_v1 if version == 1 else labels_v2))
    
    return audio, label


# Change audio to spectrogram and label to one-hot encoded label
def get_spectogram(audio, label, version, frame_length, frame_step):
    audio = tf.cast(audio, tf.float32)
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(spectrogram)

    return spectrogram, label


def get_mel_spectogram(spectrogram, label, mel_bands):
    lower_edge_hertz, upper_edge_hertz = 0, FREQUENCY // 2
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mel_bands,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=FREQUENCY,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    # mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return mel_spectrogram, label


def compute_deltas(mfccs):
    # Pad the MFCCs at the beginning and end along the time dimension (axis 1)
    padded_mfccs = tf.pad(mfccs, paddings=[[1, 1], [0, 0]], mode='SYMMETRIC')
    # Compute the deltas (first-order differences)
    deltas = padded_mfccs[2:, :] - padded_mfccs[:-2, :]
    return deltas / 2.0


def get_mfccs(mel_spectrogram, label, num_coefficients=13):
    # add delta and delta-delta
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    deltas = compute_deltas(mfccs)
    delta_deltas = compute_deltas(deltas)
    # Concatenate along the last dimension
    mfccs = tf.concat([mfccs, deltas, delta_deltas], axis=-1)
    return mfccs, label


def insure_falling_between_1_and_m1(example, label):
    max_val = tf.reduce_max(tf.abs(example), axis=-1)
    return example / max_val, label




def convert_to_ragged(example, label):
    return tf.RaggedTensor.from_tensor(example, padding=0), label


def get_last_dimension(type, num_coefficients, mel_bands, frame_length):
    if type == "spec":
        return frame_length // 2 + 1
    if type == "mel":
        return mel_bands
    if type == "mfccs":
        return num_coefficients * 3


def convert_to_tensor(example_ragged, label, num_coefficients, mel_bands, frame_length, type):
    # Determine the last dimension size based on the type of features
    last_dim_size = get_last_dimension(type, num_coefficients, mel_bands, frame_length)
    
    # Convert the ragged tensor to a fixed-size tensor
    # The shape argument is set to [None, None, last_dim_size] which means:
    # - Keep the existing sizes for the first two dimensions (batch and time steps)
    # - Set the last dimension to the calculated last_dim_size
    example_tensor = example_ragged.to_tensor(shape=[None, None, last_dim_size])

    return example_tensor, label




def get_tf_dataset(
    raw_dataset,
    frame_length,
    frame_step,
    version,
    type,
    batch_size,
    mel_bands,
    num_coefficients
):
    # Always happens
    ds = raw_dataset.map(
        lambda x: get_audio_and_label(x, version),
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda audio, label: get_spectogram(audio, label, version, frame_length, frame_step),
        num_parallel_calls=tf.data.AUTOTUNE)
    
    # Only on the occurence of the type
    if type != "spec":
        ds = ds.map(
            lambda spectrogram, label: get_mel_spectogram(spectrogram, label, mel_bands),
            num_parallel_calls=tf.data.AUTOTUNE)
    if type != "mel":
        ds = ds.map(
            lambda mel, label: get_mfccs(mel, label, num_coefficients),
            num_parallel_calls=tf.data.AUTOTUNE)
    return ds.\
        shuffle(4098).\
        map(convert_to_ragged, num_parallel_calls=tf.data.AUTOTUNE).\
        ragged_batch(batch_size).\
        map(lambda example, label: convert_to_tensor(example, label, num_coefficients, mel_bands, frame_length, type), num_parallel_calls=tf.data.AUTOTUNE).\
        prefetch(tf.data.AUTOTUNE)



# Will return three tf.data.Dataset objects, one for each split (train, validation, test)
def get_datasets(
    batch_size,
    frame_length=256,
    frame_step=128,
    version=1, # Could be 1 or 2
    type="mfccs", # Could be spec, mel, or mfcc
    mel_bands=40,
    num_coefficients=13,
):
    combined_ds = tfds.load(f'huggingface:speech_commands/v0.0{version}')
    return (
        get_tf_dataset(
            combined_ds["train"],
            frame_length,
            frame_step,
            version,
            type,
            batch_size,
            mel_bands,
            num_coefficients
        ),
        get_tf_dataset(
            combined_ds["validation"],
            frame_length,
            frame_step,
            version,
            type,
            batch_size,
            mel_bands,
            num_coefficients
        ),
        get_tf_dataset(
            combined_ds["test"],
            frame_length,
            frame_step,
            version,
            type,
            batch_size,
            mel_bands,
            num_coefficients
        ),
    )