#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
import tensorflow as tf

def attention(inputs, attention_size, time_major=False, return_alphas=False):  # def attention(datanum, attention_size, time_major=False, return_alphas=False):
    """
​
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article
​
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
​
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
​
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def BiGRU_with_Attention(datanum, max_time, n_input, gru_size, attention_size, keep_prob, weights_1, biases_1, weights_2, biases_2): # def BiGRU_with_Attention(Input, max_time, n_input, gru_size, attention_size, keep_prob, weights_1, biases_1, weights_2, biases_2):
    '''
​
    Args:
        Input: The reshaped input EEG signals
        max_time: The unfolded time slice of BiGRU Model
        n_input: The input signal size at one time
        gru_size: The number of RNN units inside the BiGRU Model
        keep_prob: The Keep probability of Dropout
        weights_1: The Weights of first fully-connected layer
        biases_1: The biases of first fully-connected layer
        weights_2: The Weights of second fully-connected layer
        biases_2: The biases of second fully-connected layer
​
    Returns:
        FC_2: Final prediction of BiGRU Model
        FC_1: Extracted features from the first fully connected layer
        alphas: Attention Weights - Studied Attention Weights
​
    '''

    # Input EEG signals
    Input = tf.reshape(Input, [-1, max_time, n_input])

    # Forward and Backward GRU models (BiGRU Models)
    gru_fw_cell = tf.contrib.rnn.GRUCell(num_units=gru_size)
    gru_bw_cell = tf.contrib.rnn.GRUCell(num_units=gru_size)

    # Dropout for the BiGRU Model
    gru_fw_drop = tf.contrib.rnn.DropoutWrapper(cell=gru_fw_cell, input_keep_prob=keep_prob)
    gru_bw_drop = tf.contrib.rnn.DropoutWrapper(cell=gru_bw_cell, input_keep_prob=keep_prob)

    # One layer Attention-based BiGRU Model
    outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(gru_fw_drop, gru_bw_drop, Input, dtype=tf.float32)

    # Attention Mechanism
    attention_output, alphas = attention(inputs=outputs, attention_size=attention_size, return_alphas=True)
    attention_output_drop = tf.nn.dropout(attention_output, keep_prob)

    # First fully-connected layer
    FC_1 = tf.matmul(attention_output_drop, weights_1) + biases_1
    FC_1 = tf.layers.batch_normalization(FC_1, training=True)
    FC_1 = tf.nn.softplus(FC_1)
    FC_1 = tf.nn.dropout(FC_1, keep_prob)

    # Second fully-connected layer
    FC_2 = tf.matmul(FC_1, weights_2) + biases_2
    FC_2 = tf.nn.softmax(FC_2)
    
    return FC_2, alphas # return FC_2, FC_1, alphas