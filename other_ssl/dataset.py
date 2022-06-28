import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def prepare_dataset(x, batch_size):
    # x_train:x_test=a:b
    a,b=10,0
    l=len(x)
    print(l)
    x_train = x[:int(l*(a/10))]
    x_test = x[int(l*(a/10)):]
    x_train = x_train[:(len(x_train)//batch_size) * batch_size]
    x_test = x_test[:(len(x_test) // batch_size) * batch_size]
    print(x_train.shape,x_test.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=3000).batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # print(tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
    test_dataset = test_dataset.shuffle(buffer_size=500).batch(batch_size, drop_remainder=True)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # print(tf.data.experimental.AUTOTUNE)

    return train_dataset,test_dataset

if __name__ == '__main__':
    x_train = np.load('/home/dell/python/Cherry_unsup/simsiam/x.npy')
    print(x_train.shape)
    batch_size=32
    train_dataset, test_dataset = prepare_dataset(x_train, batch_size)