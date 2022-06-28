import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from dataset import prepare_dataset
from augmentations import RandomResizedCrop, RandomColorJitter
from algorithms import SimCLR, NNCLR, DCCLR, BarlowTwins, HSICTwins, TWIST, MoCo, DINO

tf.get_logger().setLevel("WARN")  # suppress info-level logs

import numpy as np
from tensorflow.keras.layers import BatchNormalization, Input, Flatten, Dense, Activation, GlobalAveragePooling2D, \
    subtract

# hyperparameters
num_epochs = 100
batch_size = 48
width = 256  # 128

# hyperparameters corresponding to each algorithm
hyperparams = {
    SimCLR: {"temperature": 0.1},
    NNCLR: {"temperature": 0.1, "queue_size": 10000},
    DCCLR: {"temperature": 0.1},
    BarlowTwins: {"redundancy_reduction_weight": 10.0},
    HSICTwins: {"redundancy_reduction_weight": 3.0},
    TWIST: {},
    MoCo: {"momentum_coeff": 0.99, "temperature": 0.1, "queue_size": 10000},
    DINO: {"momentum_coeff": 0.9, "temperature": 0.1, "sharpening": 0.5},
}

# data
x_train = np.load('/home/dell/python/Cherry_unsup/simsiam/x_all_un_normalization.npy')
print(x_train.shape)
train_dataset, test_dataset = prepare_dataset(x_train, batch_size)

# encoder model
from nets.resnet50 import ResNet50


def get_encoder():
    # Input and backbone.
    inputs = layers.Input((288, 512, 3))
    feat1, feat2, feat3, feat4, feat5 = ResNet50(inputs)
    x = Flatten()(feat5)
    # x = Dense(512)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    outputs = Dense(256)(x)
    # outputs=x
    return tf.keras.Model(inputs, outputs, name="encoder")


# select an algorithm
Algorithm = MoCo
# architecture
model = Algorithm(
    contrastive_augmenter=keras.Sequential(
        [
            layers.Input(shape=(288, 512, 3)),
            preprocessing.Rescaling(1 / 255),
            preprocessing.RandomFlip("horizontal"),
            RandomResizedCrop(scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)),
            RandomColorJitter(brightness=0.7, contrast=0.6, saturation=0.6, hue=0.1),
        ],
        name="contrastive_augmenter",
    ),
    classification_augmenter=keras.Sequential(
        [
            layers.Input(shape=(288, 512, 3)),
            preprocessing.Rescaling(1 / 255),
            preprocessing.RandomFlip("horizontal"),
            RandomResizedCrop(scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
            RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ],
        name="classification_augmenter",
    ),
    encoder=get_encoder(),
    projection_head=keras.Sequential(
        [
            layers.Input(shape=(width,)),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dense(width),
            layers.BatchNormalization(),
            #
            # layers.Input(shape=(width,)),
            # layers.Dense(width, activation="relu"),
            # layers.Dense(width),
        ],
        name="projection_head",
    ),
    linear_probe=keras.Sequential(
        [
            layers.Input(shape=(width,)),
            layers.Dense(10),
        ],
        name="linear_probe",
    ),
    **hyperparams[Algorithm],
)

# optimizers
model.compile(
    contrastive_optimizer=keras.optimizers.Adam(),
    probe_optimizer=keras.optimizers.Adam(),
)

# run training
# model.summary()
history = model.fit(train_dataset, epochs=num_epochs, batch_size=batch_size)

# # save history
# with open("{}.pkl".format(Algorithm.__name__), "wb") as write_file:
#     pickle.dump(history.history, write_file)
weight_h5_name = Algorithm.__name__ + '_' + str(num_epochs) + '_' + str(batch_size) + '.h5'
model.get_layer(name='encoder').save_weights(weight_h5_name)
