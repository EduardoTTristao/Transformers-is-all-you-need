import pandas as pd
import numpy as np
import tensorflow as tf

BUFFER_SIZE = 20000
BATCH_SIZE = 16


def get_ds():
    content_train_x = open(r"../English-german/train-x", "r")
    content_train_y = open(r"../English-german/train-y", "r")

    content_train_x = content_train_x.read().split("\n")
    content_train_y = content_train_y.read().split("\n")

    df = pd.DataFrame({'x': content_train_x, 'y': content_train_y})

    df['x'] = np.array(df['x'])

    df['y'] = np.array(df['y'])

    ds = tf.data.Dataset.from_tensor_slices((df['x'], df['y']))

    return ds


def prepare_dt(tokenizer_in, tokenizer_out, ds):
    def encode(inp, out):
        inp = [tokenizer_in.vocab_size] + tokenizer_in.encode(
            inp.numpy()) + [tokenizer_in.vocab_size + 1]

        out = [tokenizer_out.vocab_size] + tokenizer_out.encode(
            out.numpy()) + [tokenizer_out.vocab_size + 1]

        return inp, out

    def tf_encode(inp, out):
        result_in, result_out = tf.py_function(encode, [inp, out], [tf.int64, tf.int64])
        result_in.set_shape([None])
        result_out.set_shape([None])

        return result_in, result_out

    MAX_LENGTH = 1115

    def filter_max_length(x, y, max_length=MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length,
                              tf.size(y) <= max_length)

    dataset = ds.map(tf_encode)

    dataset = dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    input_vocab_size = tokenizer_in.vocab_size + 2
    target_vocab_size = tokenizer_out.vocab_size + 2

    return dataset, input_vocab_size, target_vocab_size