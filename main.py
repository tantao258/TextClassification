import os
import time
import tensorflow as tf
import datetime
from utils import load_data_and_labels, batch_iter
from model import TextCNN

# Parameters
tf.app.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.app.flags.DEFINE_string("positive_data_file", "./data/chinese/pos.txt", "Data source for the positive data.")
tf.app.flags.DEFINE_string("negative_data_file", "./data/chinese/neg.txt", "Data source for the negative data.")
tf.app.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.app.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.app.flags.DEFINE_integer("evaluate_every", 20, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 30, "Save model after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
FLAGS = tf.app.flags.FLAGS

# Load data
print("Loading data...")
x_train, y_train, x_dev, y_dev, vocab_processor = load_data_and_labels(pos="./data/chinese/pos.txt",
                                                                       neg="./data/chinese/neg.txt",
                                                                       dev_sample_percentage=FLAGS.dev_sample_percentage)
# Generate batches
batches = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

cnn = TextCNN(sequence_length=x_train.shape[1],
              num_classes=y_train.shape[1],
              vocab_size=len(vocab_processor.vocabulary_),
              embedding_size=FLAGS.embedding_dim,
              filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
              num_filters=FLAGS.num_filters,
              l2_reg_lambda=FLAGS.l2_reg_lambda)


with tf.Session() as sess:

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # define summary
    grad_summaries = []
    for g, v in cnn.grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

    # merge all the train summary
    train_summary_merged = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "train"), graph=sess.graph)

    # merge all the dev summary
    dev_summary_merged = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "dev"), graph=sess.graph)

    # checkPoint saver
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # Write vocabulary
    vocab_processor.save(os.path.join(out_dir, "vocab"))

    sess.run(tf.global_variables_initializer())

    for batch in batches:
        # train loop
        x_batch, y_batch = zip(*batch)

        _, step, train_summaries, loss, accuracy = sess.run([cnn.train_op, cnn.global_step, train_summary_merged, cnn.loss, cnn.accuracy],
                                                            feed_dict={cnn.input_x: x_batch,
                                                                       cnn.input_y: y_batch,
                                                                       cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
                                                            )
        train_summary_writer.add_summary(train_summaries, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        # validation
        current_step = tf.train.global_step(sess, cnn.global_step)

        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            step, dev_summaries, loss, accuracy = sess.run([cnn.global_step, dev_summary_merged, cnn.loss, cnn.accuracy],
                                                           feed_dict={cnn.input_x: x_dev,
                                                                      cnn.input_y: y_dev,
                                                                      cnn.dropout_keep_prob: 1.0}
                                                           )
            dev_summary_writer.add_summary(dev_summaries, step)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))