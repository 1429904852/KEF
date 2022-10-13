import tensorflow as tf


def softmax_with_len(inputs, length, max_len):
    inputs = tf.cast(inputs, tf.float32)
    inputs = tf.exp(inputs)
    length = tf.reshape(length, [-1])
    mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
    inputs *= mask
    _sum = tf.reduce_sum(inputs, reduction_indices=-1, keep_dims=True) + 1e-9
    alpha = tf.div(inputs, _sum)
    return alpha


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape), dtype=tf.float32)
        gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
        normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
        outputs = gamma * normalized + beta
    return outputs


def multi_head_attention(keys, values, querys, num_units, num_heads, scope='multihead_attention', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        Q = tf.nn.relu(
            tf.layers.dense(querys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
        K = tf.nn.relu(
            tf.layers.dense(keys, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
        V = tf.nn.relu(
            tf.layers.dense(values, num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(querys)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

        outputs = tf.nn.softmax(outputs)

        query_masks = tf.sign(tf.abs(tf.reduce_sum(querys, axis=-1)))
        query_masks = tf.tile(query_masks, [num_heads, 1])
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
        outputs *= query_masks
        # [-1, 32, 49]
        # print(outputs.shape)

        outputs = tf.matmul(outputs, V_)

        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += querys
        outputs = normalize(outputs)

        u1 = tf.layers.dense(outputs, num_units, use_bias=True)  # (N, T_q, C)
        u2 = tf.nn.relu(u1)
        outputs = tf.layers.dense(u2, num_units, use_bias=True)

    return outputs


def bilinear_attention_layer(output_layer, output_target_layer, sequence_len, hidden_size, scope_name):
    att_weight = tf.get_variable('att_sen' + scope_name, [hidden_size, hidden_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
    # target
    output_target_sen = tf.expand_dims(output_target_layer, 2)
    # target attention sentence
    inputs = tf.reshape(output_layer, [-1, hidden_size])
    tmp = tf.reshape(tf.matmul(inputs, att_weight), [-1, sequence_len, hidden_size])
    tmp = tf.reshape(tf.matmul(tmp, output_target_sen), [-1, 1, sequence_len])
    tmp = tf.cast(tmp, tf.float32)
    tmp = tf.exp(tmp)
    _sum = tf.reduce_sum(tmp, reduction_indices=-1, keep_dims=True) + 1e-9
    alpha = tf.div(tmp, _sum, name='attention_sen' + scope_name) + 1e-9
    output_layer = tf.reshape(tf.squeeze(tf.matmul(alpha, output_layer)), [-1, hidden_size])

    return output_layer


def bilinear_attention_layer_1(output_layer, output_target_layer, sequence_len, hidden_size, img_demension, scope_name):
    att_weight = tf.get_variable('att_sen' + scope_name, [hidden_size, img_demension],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
    # target
    # [-1, 2048, 1]
    output_target_sen = tf.expand_dims(output_target_layer, 2)
    # target attention sentence
    # [-1, 20, 768]
    inputs = tf.reshape(output_layer, [-1, hidden_size])
    # [-1, 20, 2048]
    tmp = tf.reshape(tf.matmul(inputs, att_weight), [-1, sequence_len, img_demension])
    tmp = tf.reshape(tf.matmul(tmp, output_target_sen), [-1, 1, sequence_len])
    tmp = tf.cast(tmp, tf.float32)
    # tmp = tf.exp(tmp)
    _sum = tf.reduce_sum(tmp, reduction_indices=-1, keep_dims=True) + 1e-6
    alpha = tf.div(tmp, _sum, name='attention_sen' + scope_name) + 1e-6
    output_layer = tf.reshape(tf.squeeze(tf.matmul(alpha, output_layer)), [-1, hidden_size])

    return output_layer


def reduce_mean_with_len(inputs, length):
    """
    :param inputs: 3-D tensor
    :param length: the length of dim [1]
    :return: 2-D tensor
    """
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
    return inputs


def last_mean_with_len(outputs, n_hidden, length, max_len):
    batch_size = tf.shape(outputs)[0]
    index = tf.range(0, batch_size) * max_len + (length - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)  # batch_size * 2n_hidden+1
    return outputs


def mlp_attention_layer(inputs, length, n_hidden, l2_reg, random_base):
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    w = tf.get_variable(
        name='att_w_',
        shape=[n_hidden, n_hidden],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    u = tf.get_variable(
        name='att_u_',
        shape=[n_hidden, 1],
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.matmul(inputs, w)
    tmp = tf.reshape(tf.matmul(tmp, u), [batch_size, 1, max_len])
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha


def margin_loss(y_true, y_pred):
    """
    :param y_true: [None, n_classes]
    :param y_pred: [None, n_classes]
    :return: a scalar loss value.
    """
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    assert_inf_L = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_inf(L))), ['assert_inf_L', L], summarize=100)
    assert_nan_L = tf.Assert(tf.logical_not(tf.reduce_any(tf.is_nan(L))), ['assert_nan_L', L], summarize=100)
    with tf.control_dependencies([assert_inf_L, assert_nan_L]):
        ret = tf.reduce_mean(tf.reduce_sum(L, axis=1))
    return ret
