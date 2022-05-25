from tensorflow.keras import layers
from tensorflow import keras

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


def conv_base_model(data_config, exp_config, classes):
    """
    function that generates the base model for optimization-based meta-learning experiments
    :param data_config: configuration file with dataset information
    :param exp_config: configuration file for specific experiment information
    :param classes: number of wayw of the experiment
    :return: generated base model
    """

    x_size_image = data_config["x_size_image"]
    y_size_image = data_config["y_size_image"]
    channels = data_config["image_channels"]
    conv_filters_number = exp_config["conv_filters_per_layer"]
    conv_kernel_size = exp_config["conv_kernel_size"]
    inputs = layers.Input(shape=(x_size_image, y_size_image, channels))

    x = conv_bn(inputs, filt_num=round(conv_filters_number / 2), kern_size=conv_kernel_size)
    x = conv_bn(x, filt_num=conv_filters_number, kern_size=conv_kernel_size)
    x = conv_bn(x, filt_num=round(conv_filters_number * 2), kern_size=conv_kernel_size)

    flatten = layers.Flatten()(x)

    outputs = layers.Dense(classes, activation="softmax")(flatten)
    base_model = keras.Model(inputs=inputs, outputs=outputs)

    return base_model
