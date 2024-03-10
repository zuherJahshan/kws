import tensorflow as tf


class StateTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        projection_dim,
        inner_ff_dim,
        dropout=0.0,
        kernel_regularizer=None,
    ):
        super(StateTransformerBlock, self).__init__()
        # primitive properties
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        
        # layers
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
        )
        self.add1 = tf.keras.layers.Add()
        self.layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.inner_dense = tf.keras.layers.Dense(
            units=inner_ff_dim,
            kernel_regularizer=kernel_regularizer,
            activation="relu",
        )
        self.outer_dense = tf.keras.layers.Dense(
            units=projection_dim,
            kernel_regularizer=kernel_regularizer,
        )
        self.ff_dropout = tf.keras.layers.Dropout(dropout)
        self.add2 = tf.keras.layers.Add()
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, state_seq, input_seq):
        # state sequence is of shape (batch_size, num_of_state_cells, projection_dim)
        # input sequence is of shape (batch_size, input_size, projection_dim)
        store_seq = tf.concat([state_seq, input_seq], axis=1)
        attention_output = self.attention(state_seq, store_seq)
        attention_output = self.add1([attention_output, state_seq])
        attention_output = self.layernorm_1(attention_output)
        inner_output = self.inner_dense(attention_output)
        outer_output = self.outer_dense(inner_output)
        outer_output = self.ff_dropout(outer_output)
        outer_output = self.add2([outer_output, attention_output])
        return self.layernorm_2(outer_output) # the output is of shape (batch_size, num_of_state_cells, projection_dim)
    

class StateTransformer(tf.keras.models.Model):
    def __init__(
        self,
        num_classes,
        num_heads,
        num_state_cells,
        input_seq_size,
        projection_dim,
        inner_ff_dim,
        dropout=0.0,
        kernel_regularizer=None,
    ):
        super(StateTransformer, self).__init__()
        # primitive properties
        self.projection_dim = projection_dim
        self.num_state_cells = num_state_cells
        self.input_seq_size = input_seq_size

        self.encoding = tf.keras.layers.Dense(
            units=projection_dim,
            kernel_regularizer=kernel_regularizer,
        )
        # State TE layers
        self.calc_z = StateTransformerBlock(
            num_heads=num_heads,
            projection_dim=projection_dim,
            inner_ff_dim=inner_ff_dim,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
        )
        self.calc_r = StateTransformerBlock(
            num_heads=num_heads,
            projection_dim=projection_dim,
            inner_ff_dim=inner_ff_dim,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
        )
        self.calc_current_state = StateTransformerBlock(
            num_heads=num_heads,
            projection_dim=projection_dim,
            inner_ff_dim=inner_ff_dim,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
        )
        self.classifier = tf.keras.layers.Dense(
            units=num_classes,
            kernel_regularizer=kernel_regularizer,
            activation="softmax",
        )

    def call(self, input_seq):
        # Assume the input sequence is of the shape (batch_size, all_seq, input_seq), we want to reshape it to be (batch_size, -1, input_seq_size, projection_dim)
        input_seq = self.encoding(input_seq)
        # initialize the state sequence
        batch_size = tf.shape(input_seq)[0]
        state_t = tf.zeros([batch_size, self.num_state_cells, self.projection_dim])
        folds = tf.shape(input_seq)[1] // self.input_seq_size
        for fold in range(folds):
            curr_input_seq = input_seq[:, fold*self.input_seq_size:(fold+1)*self.input_seq_size, :]
            # Pad in case values are missing
            if tf.shape(curr_input_seq)[1] < self.input_seq_size:
                curr_input_seq = tf.pad(curr_input_seq, [[0, 0], [0, self.input_seq_size - tf.shape(curr_input_seq)[1]], [0, 0]])
            z = tf.keras.activations.sigmoid(self.calc_z(state_t, curr_input_seq))
            r = tf.keras.activations.sigmoid(self.calc_r(state_t, curr_input_seq))
            current_state = self.calc_current_state(r*state_t, curr_input_seq)
            state_t = (1 - z)*state_t + z*current_state
        
        return self.classifier(state_t[:, 0, :])

        