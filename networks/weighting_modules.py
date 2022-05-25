from abc import ABC
import tensorflow as tf


class Embedding_Module(tf.keras.Model, ABC):
    """The Embedding Module takes as distinct input the support and query
    examples and generate an embedding for the following comparison phase.
    The training of this module is only performed on the support data for
    the pure few shot learning."""

    def __init__(self, feature_dimension):
        super(Embedding_Module, self).__init__()
        self.layer1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding="valid"),  # kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2)])
        self.layer2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 6), padding="valid"),  # kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2)])
        self.layer3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(6, 7), padding="valid"),
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU()])
        self.layer4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=feature_dimension, kernel_size=(7), padding="valid"),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(name="embedding")])

    def call(self, inp):
        # input shape: (?, 105, 105, 1)
        out = self.layer1(inp)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        print("OUT EMBB:", out.shape)
        return out  # (?, 18, 18, 32)


class Comparison_Module(tf.keras.Model, ABC):
    """The Comparison module take as input the channel-wise concatenation of support and query samples.
    It enables the extraction of mixed features. For N available support per class,
    the generated comparison vectors are averaged over N"""

    def __init__(self, feature_dimension, classes):
        super(Comparison_Module, self).__init__()
        self.feature_dimension = feature_dimension
        self.classes = classes
        self.layer1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=feature_dimension, kernel_size=(3, 1), strides=(3, 1), padding="valid"),
            # kernel_initializer=initializer),
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(3)])
        self.layer2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=feature_dimension, kernel_size=3, padding="valid"),
            # kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(momentum=1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D()
        ])

    def call(self, inp, num_train_shots):
        print("Concatenated Examples: ", inp.shape)
        out = self.layer1(inp)
        out = self.layer2(out)
        print("B SHAPE: ", out.shape)
        out = tf.reshape(out, [-1, self.classes, num_train_shots, self.feature_dimension])
        out = tf.math.reduce_mean(out, axis=2, keepdims=False)
        out = tf.reshape(out, [-1, self.classes * self.feature_dimension])
        print("C SHAPE: ", out.shape)
        return out


class Weighting_Module(tf.keras.Model, ABC):
    """The Weighting Module takes as input the comparison vectors and generates
    a distribution of probabilities over the available classes"""

    def __init__(self, weighting_dim, classes):
        super(Weighting_Module, self).__init__()
        self.fc1 = tf.keras.layers.Dense(weighting_dim, activation="relu")  # kernel_initializer=initializer)
        self.fc2 = tf.keras.layers.Dense(classes, activation="softmax")  # kernel_initializer=initializer)

    def call(self, inp):
        print("Weighting Examples: ", inp.shape)
        out = self.fc1(inp)
        out = self.fc2(out)
        print("E SHAPE: ", out.shape)
        return out


class Full_Pipeline(tf.keras.Model, ABC):
    """The Full Pipeline handles the information flow for Embedding, Comparison and Weighting Modules"""

    def __init__(self, classes, feature_dimension, weighting_dim):
        super(Full_Pipeline, self).__init__()
        self.classes = classes
        self.feature_dimension = feature_dimension
        self.weighting_dim = weighting_dim
        self.embedding_model = Embedding_Module(self.feature_dimension)
        self.weighting_model = Weighting_Module(self.weighting_dim, self.classes)
        self.comparison_model = Comparison_Module(self.feature_dimension, self.classes)
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, support_samples, query_input, num_shots_tr, num_shots_qr_ts, batch_s=None, multi_query=True):

        if batch_s is None:
            batch_s = self.classes

        support_features = self.embedding_model(support_samples)

        print(support_features.shape)
        if multi_query:
            # if more than 1 query, the support have to be replicated N times respect to the number of query shots
            support_features_ext = tf.repeat(tf.expand_dims(support_features, 0), num_shots_qr_ts * batch_s, axis=0)
        else:
            # single prediction
            support_features_ext = tf.expand_dims(support_features, 0)
        # print("extended dim support: ", support_features_ext.shape)
        query_features = tf.stop_gradient(self.embedding_model(query_input))  # no backprop here
        # extend the query for the number of classes (1-support each class)
        query_features_ext = tf.repeat(tf.expand_dims(query_features, 0), num_shots_tr * self.classes, axis=0)
        query_features_ext = tf.experimental.numpy.moveaxis(query_features_ext, 0, 1)
        print("extended query: ", query_features_ext.shape)
        comparison_features = self.concat([support_features_ext, query_features_ext])
        comparison_features = tf.reshape(comparison_features, [-1, self.feature_dimension * 2, 12, 12])
        print(comparison_features.shape)
        weighting_features = self.comparison_model(comparison_features, num_shots_tr)

        out = self.weighting_model(weighting_features)
        print("FINAL: ", out.shape)
        input()
        return out
