from abc import ABC

import tensorflow as tf


def conv_bn(filter_num=128, kern_size=3, stride=2):
    """
    Definition of the single convolutional layer for the model in the internal loop
    :param filter_num: number of filters for the current layer
    :param kern_size: size of the convolutional kernel
    :param stride: size of the convolutional stride for the layer

    :return: ReLU of the new layer
    """
    # normal all 128
    layer = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=filter_num, kernel_size=kern_size, strides=stride, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()])
    return layer


class Conv_Dense_Module(tf.keras.Model, ABC):
    """
    class that implements all the function for the base model in the optimization-based meta-learning experiments

    """
    def __init__(self, n_ways, conv_filters_number, conv_kernel_size):
        """
        function that generates the base model for optimization-based meta-learning experiments
        :param n_ways: number of ways for the classification experiment
        :param conv_filters_number: number of convolution filters of the second layer
        :param conv_kernel_size: kernel size for the convolutional layers
        :return: generated base model
        """
        super(Conv_Dense_Module, self).__init__()

        # inputs = tf.keras.layers.Input(shape=(x_size_image, y_size_image, channels))
        self.layer1 = conv_bn(filter_num=round(conv_filters_number / 2), kern_size=conv_kernel_size)
        self.layer2 = conv_bn(filter_num=conv_filters_number, kern_size=conv_kernel_size)
        self.layer3 = conv_bn(filter_num=round(conv_filters_number * 2), kern_size=conv_kernel_size)
        self.flatten = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(n_ways, activation="softmax")

    def call(self, inp):
        # input shape: (?, dataset_config["x_size_image"], dataset_config["y_size_image"],
        # dataset_config["image_channels"])
        out = self.layer1(inp)
        out = self.layer2(out)
        out = self.layer3(out)
        flat = self.flatten(out)
        predict = self.out(flat)

        return predict
