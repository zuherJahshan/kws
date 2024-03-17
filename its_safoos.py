import tensorflow as tf


class GatedMlpBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        inner_dim,
        outer_dim,
        non_linearity,
    ):
        super(GatedMlpBlock, self).__init__()
        self.inner_dense_non_linear = tf.keras.layers.Dense(
            units=inner_dim,
            activation=non_linearity,
        )
        self.inner_dense_linear = tf.keras.layers.Dense(
            units=inner_dim,
        )
        self.outer_dense = tf.keras.layers.Dense(
            units=outer_dim,
        )

    def call(self, input_seq):
        inner_non_linear = self.inner_dense_non_linear(input_seq)
        inner_linear = self.inner_dense_linear(input_seq)
        multiply = inner_non_linear * inner_linear
        return self.outer_dense(multiply)


class MultiQueryAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        proj_dim,
        dropout=0.0,
        kernel_regularizer=None,
    ):
        super(MultiQueryAttention, self).__init__()
        
        # define linear layers for key and value
        self.key_layer = tf.keras.layers.Dense(
            units=proj_dim,
            kernel_regularizer=kernel_regularizer,
        )
        self.value_layer = tf.keras.layers.Dense(
            units=proj_dim,
            kernel_regularizer=kernel_regularizer,
        )

        # define linear layers for query, as the number of heads
        self.query_layers = [tf.keras.layers.Dense(
            units=proj_dim,
            kernel_regularizer=kernel_regularizer,
        ) for _ in range(num_heads)]

        # define linear layer for output
        self.output_layer = tf.keras.layers.Dense(
            units=proj_dim,
            kernel_regularizer=kernel_regularizer,
        )


    def _compute_attn(
        self,
        query, # shape will be [B,S,d]
        input_keys, # shape will be [B,T,d]
        memory_keys, # shape will be [B,S,d]
        input_vals, # shape will be [B,T,d]
        memory_vals, # shape will be [B,S,d]
    ):
        # Assume S represents the number of memory cells and T represents the number of input cells
        # Compute the attention weights
        
        # Compute the score a memory cell gives to an input cell
        input_score = tf.matmul(query, input_keys, transpose_b=True)
        # Shape will be [B,S,T]. This will result in a matrix,
        # s.t. row i describes how much attention should the query i give all other input cells
        
        self_score = query * memory_keys
        self_score = tf.reduce_sum(self_score, axis=-1, keepdims=True)
        # Shape will be [B,S,1]. This will result in a vector,
        # s.t. element i describes how much attention should the query i give to itself

        # Concat self_score with input_score
        score = tf.concat([self_score, input_score], axis=-1)
        # Shape will be [B,S,T+1]. This will result in a matrix,
        # s.t. row i describes how much attention should the query i give to inputs and itself

        score /= tf.math.sqrt(tf.cast(tf.shape(input_keys)[-1], tf.float32))
        attn = tf.nn.softmax(score, axis=-1)

        # Break attn to [B,S,1] and [B,S,T]
        self_attn = attn[:, :, 0:1]
        input_attn = attn[:, :, 1:]

        value_of_input = tf.matmul(input_attn, input_vals) # shape will be [B,S,d]
        value_of_self = self_attn * memory_vals # shape will be [B,S,d]
        return value_of_input + value_of_self


    def call(self, input_seq, memory_cells):
        # query_seq is of shape (batch_size, input_size, key_dim)
        # store_seq is of shape (batch_size, store_seq, key_dim)
        # compute the attention weights
        ik = self.key_layer(input_seq)
        mk = self.key_layer(memory_cells)
        iv = self.value_layer(input_seq)
        mv = self.value_layer(memory_cells)
        attns = [self._compute_attn(q, ik, mk, iv, mv) for q in [layer(memory_cells) for layer in self.query_layers]]
        concat = tf.concat(attns, axis=-1)
        return self.output_layer(concat)
        

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
        self.attention = MultiQueryAttention(
            num_heads=num_heads,
            proj_dim=projection_dim,
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
        self.outer_dense = GatedMlpBlock(
            inner_dim=inner_ff_dim,
            outer_dim=projection_dim,
            non_linearity="relu",
        )
        self.ff_dropout = tf.keras.layers.Dropout(dropout)
        self.add2 = tf.keras.layers.Add()
        self.layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, state_seq, input_seq):
        # state sequence is of shape (batch_size, num_of_state_cells, projection_dim)
        # input sequence is of shape (batch_size, input_size, projection_dim)
        # store_seq = tf.concat([state_seq, input_seq], axis=1)
        attention_output = self.attention(input_seq, state_seq)
        attention_output = self.add1([attention_output, state_seq])
        attention_output = self.layernorm_1(attention_output)
        inner_output = self.inner_dense(attention_output)
        outer_output = self.outer_dense(inner_output)
        outer_output = self.ff_dropout(outer_output)
        outer_output = self.add2([outer_output, attention_output])
        return self.layernorm_2(outer_output) # the output is of shape (batch_size, num_of_state_cells, projection_dim)
    

class ITSRU(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        num_state_cells,
        projection_dim,
        inner_ff_dim,
        initial_state_trainability=False,
        dropout=0.0,
        kernel_regularizer=None,
    ):
        super(ITSRU, self).__init__()

        self.encoding = tf.keras.layers.Dense(
            units=projection_dim,
            kernel_regularizer=kernel_regularizer,
        )
        # Initialize the learnable initial state
        self.initial_state = self.add_weight(
            shape=(1, num_state_cells, projection_dim),
            initializer='random_normal',
            trainable=initial_state_trainability,
            name='initial_state'
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


    def set_initial_state_trainability(self, trainable):
        self.initial_state._trainable = trainable


    def call(self, input_seq):
        # Assume that input is of size [B,T,S,D] where B is the batch size, T is the number of time steps, S is the sequence length at each timestep, and D is the feature dimension
        input_seq = self.encoding(input_seq)
        # initialize the state sequence
        batch_size = tf.shape(input_seq)[0]
        # Use the learnable initial state, replicate it for the whole batch
        state_t = tf.tile(self.initial_state, [batch_size, 1, 1])
        
        folds = tf.shape(input_seq)[1]
        states = tf.TensorArray(
            tf.float32,
            dynamic_size=True,
            size=0
        )
        for fold in range(folds):
            curr_input_seq = input_seq[:, fold, :, :]
            z = self.calc_z(state_t, curr_input_seq)
            r = self.calc_r(state_t, curr_input_seq)
            current_state = self.calc_current_state(r*state_t, curr_input_seq)
            state_t = (1 - z)*state_t + z*current_state
            states = states.write(fold, state_t)#.mark_used()
        
        return tf.transpose(
            states.stack(),
            [1, 0, 2, 3]
        )


class ITS(tf.keras.models.Model):
    def __init__(
        self,
        num_classes,
        num_heads,
        num_repeats,
        num_state_cells,
        input_seq_size,
        projection_dim,
        inner_ff_dim,
        initial_state_trainability=False,
        dropout=0.0,
        kernel_regularizer=None,
    ):
        super(ITS, self).__init__()
        # the input sequence size
        self.input_seq_size = input_seq_size
        
        # ITS recurrent units
        self.itsrus = [ ITSRU(
            num_heads=num_heads,
            num_state_cells=num_state_cells,
            projection_dim=projection_dim,
            inner_ff_dim=inner_ff_dim,
            initial_state_trainability=initial_state_trainability,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
        ) for _ in range(num_repeats) ]
        
        # self.label_token = self.add_weight(
        #     shape=(1, 1, projection_dim),
        #     initializer='random_normal',
        #     trainable=initial_state_trainability,
        #     name='initial_state'
        # )
        # self.mixer = StateTransformerBlock(
        #     num_heads=num_heads,
        #     projection_dim=projection_dim,
        #     inner_ff_dim=inner_ff_dim,
        #     dropout=dropout,
        #     kernel_regularizer=kernel_regularizer,
        # )

        self.classifier = tf.keras.layers.Dense(
            units=num_classes,
            activation="softmax",
        )



    def call(self, input_seq):
        # input_seq is of shape (batch_size, input_size, feature_dim).
        # First of all, we will transform it to the shape (batch_size, folds, input_seq_size, projection_dim)
        # Pad the input sequence to the nearest multiple of input_seq_size
        input_seq_size = input_seq.shape[1]
        folds = tf.cast(tf.math.ceil(input_seq_size / self.input_seq_size), tf.int32)
        final_time_steps = folds * self.input_seq_size
        input_seq = tf.pad(
            input_seq,
            [[0, 0], [0, final_time_steps - input_seq_size], [0, 0]]
        )
        
        input_seq = tf.reshape(
            input_seq,
            [-1, folds, self.input_seq_size, input_seq.shape[-1]]
        )
        # pass the input sequence through the ITSRUs
        x = input_seq
        for itsru in self.itsrus:
            x = itsru(x)

        # mix the states of the last timestep with the label token
        # transform the label weight to the shape (batch_size, 1, projection_dim)
        # label_token = tf.tile(self.label_token, [tf.shape(x)[0], 1, 1])
        # x = self.mixer(label_token, x[:, -1, 0, :])
        # x = tf.squeeze(x, axis=1)

        return self.classifier(x[:, -1, 0, :])