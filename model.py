import tensorflow as tf


class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0
                 ):

        l2_loss = tf.constant(0.0)

        with tf.name_scope("Input"):
            with tf.name_scope("Input_x"):
                self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            with tf.name_scope("Input_y"):
                self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars = tf.transpose(self.embedded_chars, perm=[0, 2, 1])
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # shape=(?, 128, 10096, 1)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # conv1 = tf.layers.conv2d(inputs=self.embedded_chars_expanded,
                #                          filters=num_filters,
                #                          kernel_size=[embedding_size, filter_size],
                #                          strides=1,
                #                          padding="valid",
                #                          data_format="channels_last",
                #                          use_bias=True)
                # conv1_relu = tf.nn.relu(conv1)
                # pooling1 = tf.layers.max_pooling2d(inputs=conv1_relu,
                #                                    pool_size=[1, sequence_length - filter_size + 1],
                #                                    strides=1,
                #                                    padding="valid")
                # pooled_outputs.append(pooling1)

                # Convolution Layer
                filter_shape = [embedding_size, filter_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(input=self.embedded_chars_expanded,
                                    filter=W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    data_format="NHWC",
                                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Pooling Layer
                pooled = tf.nn.max_pool(value=h,
                                        ksize=[1, 1, sequence_length - filter_size + 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        data_format="NHWC",
                                        name="pool")
                pooled_outputs.append(pooled)

        # combine the pooled feature
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("drop_out"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("FC"):
            W = tf.get_variable("W", shape=[num_filters_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("train"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            self.grads_and_vars = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        with tf.name_scope("prediction"):
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")