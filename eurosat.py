"""Eurosat dataset from tfds.

Convolutional neural network using pretrained base resnet50.
"""

from keras.applications import resnet50
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.initializers import HeNormal
from keras.layers import (Dense, Dropout, Flatten, IntegerLookup,
                          RandomBrightness, RandomContrast,
                          RandomFlip)
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import Adam
from keras import models
from math import floor
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def build_model(param_dict, base):
    input = keras.Input(shape=(64, 64, 3))
    
    x = RandomFlip()(input)
    x = RandomBrightness(0.5)(x)
    x = RandomContrast(0.5)(x)
    x = base(x)
    x = Flatten()(x)
    x = Dropout(0.1)(x)
    x = Dense(512, activation="relu", kernel_initializer=HeNormal())(x)
    
    output = Dense(11, activation="softmax")(x)
    
    model = keras.Model(input, output)
    model.compile(optimizer=Adam(learning_rate=param_dict["lr"]),
                  loss=CategoricalCrossentropy(),
                  metrics=[CategoricalAccuracy()])
    
    return model

def get_callbacks():
    early_stop_loss = EarlyStopping(monitor="loss", patience=8)
    reduce_lr_plateau = ReduceLROnPlateau(monitor="loss", factor=0.9,
                                          patience=4)

    return [early_stop_loss, reduce_lr_plateau]
    
def get_eurosat_dataset():
    images, labels = tfds.load("eurosat", split="train",
                               as_supervised=True, batch_size=-1)

    return images, labels

def get_random_numbers(middle):
    rng = np.random.default_rng()
    param_dict = {"lr": middle * (0.9 + 0.2 * rng.random())}
    print(param_dict)
    
    return param_dict

def plot(history):
    acc = history.history["categorical_accuracy"]
    loss = history.history["loss"]
    val_acc = history.history["val_categorical_accuracy"]
    val_loss = history.history["val_loss"]

    epochs = range(len(loss))
    start_at = 2
    
    plt.plot(epochs[start_at:], loss[start_at:], "ob",
             label="Training Loss")
    plt.plot(epochs[start_at:], val_loss[start_at:], "xb",
             label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    plt.plot(epochs[start_at:], acc[start_at:], "ob",
             label="Training Accuracy")
    plt.plot(epochs[start_at:], val_acc[start_at:], "xb",
             label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def main():
    # Load eurosat dataset as tuple of tensors: (images, labels).
    images, labels = get_eurosat_dataset()

    # Get percent indices.
    percents = [floor(np.shape(images)[0] * (i / 100.0))
                for i in range(100)]

    # Get resnet50 pretrained base and freeze layers.
    base = resnet50.ResNet50(include_top=False,
                             input_shape=(64, 64, 3))
    base.trainable = False

    # Get vocabulary for labels.
    label_vocab = [i for i in range(10)]

    # One-hot-encode labels.
    one_hot_encode_layer = IntegerLookup(vocabulary=label_vocab,
                                         output_mode="one_hot")
    encoded_labels = one_hot_encode_layer(labels)

    # Split dataset.
    split = 80
    train_images = images[:percents[split]]
    test_images = images[percents[split]:]
    train_labels = encoded_labels[:percents[split]]
    test_labels = encoded_labels[percents[split]:]

    # Get callbacks.
    callbacks = get_callbacks()

    # Fit model.
    print('Fitting model.')
    model = build_model(get_random_numbers(0.01), base)
    history = model.fit(train_images, train_labels, batch_size=128,
                        callbacks=callbacks, epochs=64, shuffle=True,
                        validation_split=0.1, verbose=1)

    plot(history)
    
    # Fine-tune by unfreezing base layers.
    print("Fine-tuning.")
    base.trainable = True
    model = build_model(get_random_numbers(0.0001), base)
    history = model.fit(train_images, train_labels, batch_size=128,
                        callbacks=callbacks, epochs=128, shuffle=True,
                        validation_split=0.1, verbose=1)

    plot(history)

    # Make predictions.
    print("Predicting.")
    model.evaluate(test_images, test_labels, verbose=1)

if __name__ == "__main__":
    main()
