import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def set_placeholders(sequence_length, num_classes):
    x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
    y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    return x, y, dropout_keep_prob


def create_embedding_layer(vocab_size, embedding_size, x):
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
        embedded_chars_expanded = tf.expand_dims(tf.nn.embedding_lookup(W, x), -1)
        return W, embedded_chars_expanded


def create_CNN(filter_sizes, embedding_size, num_filters, embedded_chars_expanded, sequence_length, dropout_keep_prob):
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                    padding='VALID', name="pool")
            pooled_outputs.append(pooled)

    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters * len(filter_sizes)])

    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    return h_drop


def get_scores_and_predictions(num_filters_total, num_classes, l2_loss, h_drop):
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.glorot_uniform_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        l2_loss += tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
        scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        predictions = tf.argmax(scores, 1, name="predictions")
        return l2_loss, scores, predictions


class TextCNN(object):

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0):
        self.input_x, self.input_y, self.dropout_keep_prob = set_placeholders(sequence_length, num_classes)

        self.W, self.embedded_chars_expanded = create_embedding_layer(vocab_size, embedding_size, self.input_x)

        self.h_drop = create_CNN(filter_sizes, embedding_size, num_filters, self.embedded_chars_expanded,
                                 sequence_length, self.dropout_keep_prob)

        l2_loss = tf.constant(0.0)
        l2_loss, self.scores, self.predictions = get_scores_and_predictions(num_filters * len(filter_sizes),
                                                                            num_classes, l2_loss, self.h_drop)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
