from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers

from utils.json_functions import read_json


def conv_bn(x_l, filt_num=128, kern_size=3, stride=2, gaussian_noise=False):
    """
    Definition of the single convolutional layer for the model in the internal loop
    :param x_l: previous layer of the network
    :param filt_num: number of filters for the current layer
    :param kern_size: size of the convolutional kernel
    :param stride: size of the convolutional stride for the layer
    :param gaussian_noise: set if the gaussian noise layer is wanted in the network

    :return: ReLU of the new layer
    """
    # normal all 128
    x_l = layers.Conv2D(filters=filt_num, kernel_size=kern_size, strides=stride, padding="same")(x_l)
    x_l = layers.BatchNormalization()(x_l)
    if gaussian_noise:
        x_l = layers.GaussianNoise(0.0)(x_l)
    return layers.ReLU()(x_l)


def conv_base_model(n_ways, conv_filters_number, conv_kernel_size):
    """
    function that generates the base model for optimization-based meta-learning experiments
    :param n_ways: number of ways for the classification experiment
    :param conv_filters_number: number of convolution filters of the second layer
    :param conv_kernel_size: kernel size for the convolutional layers
    :return: generated base model
    """
    dataset_config_file = "configurations/general_config/dataset_config.json"
    dataset_config_file = Path(dataset_config_file)

    data_config = read_json(dataset_config_file)

    # todo load directly here the needed data config parameters

    x_size_image = data_config["x_size_image"]
    y_size_image = data_config["y_size_image"]
    channels = data_config["image_channels"]

    inputs = layers.Input(shape=(x_size_image, y_size_image, channels))

    x = conv_bn(inputs, filt_num=round(conv_filters_number / 2), kern_size=conv_kernel_size)
    x = conv_bn(x, filt_num=conv_filters_number, kern_size=conv_kernel_size)
    x = conv_bn(x, filt_num=round(conv_filters_number * 2), kern_size=conv_kernel_size)

    flatten = layers.Flatten()(x)

    outputs = layers.Dense(n_ways, activation="softmax")(flatten)
    base_model = keras.Model(inputs=inputs, outputs=outputs)

    return base_model
