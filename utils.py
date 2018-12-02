import numpy as np
import codecs
from tensorflow.contrib import learn
import jieba


def load_data_and_labels(pos="./data/chinese/pos.txt",
                         neg="./data/chinese/neg.txt",
                         dev_sample_percentage=0.1
                         ):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(codecs.open(pos, "r", "utf-8").readlines())
    positive_examples = [[item for item in jieba.cut(s, cut_all=False)] for s in positive_examples]
    negative_examples = list(codecs.open(neg, "r", "utf-8").readlines())
    negative_examples = [[item for item in jieba.cut(s, cut_all=False)] for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)

    # build vocabulary
    max_document_length = max([len(x) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    text_list = []
    for text in x_text:
        text_list.append(' '.join(text))
    x = np.array(list(vocab_processor.fit_transform(text_list)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    return x_train, y_train, x_dev, y_dev, vocab_processor


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
