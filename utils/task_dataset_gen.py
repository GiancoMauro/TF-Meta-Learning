import os
import random

import matplotlib.pyplot as plt
import numpy as np


class Dataset:
    """
    # This class facilitates the creation of a few-shot dataset
    """

    def __init__(self, training, config, classes):
        """
        # Iterate over the dataset to get each individual image and its class, and put that data into a dictionary.

        :param training: if True considers the dataset as training
        :param config: configuration file with training and dataset information
        :param classes: number of ways of the experiment
        """

        self.x_size_image = config["x_size_image"]
        self.y_size_image = config["y_size_image"]
        self.channels = config["image_channels"]
        self.classes = classes
        self.dataset_name = config["name_dataset"]

        self.gen_labels = [str(lab) for lab in range(self.classes)]

        self.training_dirs = []
        self.test_dirs = []
        self.list_dirs(config["training_data_folder"], self.training_dirs)
        self.list_dirs(config["test_data_folder"], self.test_dirs)
        self.classes_tags = self.training_dirs + self.test_dirs

        if training:
            self.data_folder = config["training_data_folder"]
            self.dirs = self.training_dirs
        else: # test
            self.data_folder = config["test_data_folder"]
            self.dirs = self.test_dirs

    def list_dirs(self, root_dir, dirs_tags, subdir_levels=5):
        """
        function that generates all the sub-folders of the training/test available classes for the random subset sampling
        root_dir: main directory
        dirs_tags: list where to store all the generated classes tags for the dataset
        subdir_levels: number of subdirectory levels to the dataset classes
        (default 5: e.g. data/omniglot/training/Alphabet_of_the_Magi/character01)
        """

        for it in os.scandir(root_dir):
            if it.is_dir():
                splits = it.path.replace("\\", "/").split("/")
                if len(splits) == subdir_levels:
                    path_name = '/'.join(splits[-2:])  # Join the last two components of the split path
                    dirs_tags.append(path_name)
                self.list_dirs(it.path, dirs_tags)

    def get_mini_dataset(
            self, training_sho, num_classes, test_split=False,
            testing_sho=1, query_split=False, query_sho=1
    ):
        """
        function that generates a tensor flow "mini dataset" as set of images and respective
        labels for a given or random generated task. The generated output can be provided to a "Tf.tape" for training
        :param training_sho: number of shots per task for the training set
        :param num_classes: number of ways (classes) of the task
        :param test_split: set to True whether a test split is wanted
        :param testing_sho: number of shots per task for the test set
        :param query_split: to be set to True whether a query split is wanted
        :param query_sho: number of shots per task for the query set

        :return: tf.dataset ready for training plus images and labels for query and test sets
        """

        few_shot_train_labels = np.zeros(shape=(num_classes * training_sho))
        few_shot_train_images = np.zeros(shape=(num_classes * training_sho, self.x_size_image,
                                                self.y_size_image, self.channels))

        # not mandatory arrays:
        few_shot_test_labels = np.zeros(shape=(testing_sho * num_classes))
        few_shot_test_images = np.zeros(shape=(num_classes * testing_sho, self.x_size_image,
                                               self.y_size_image, self.channels))
        few_shot_query_labels = np.zeros(shape=(query_sho * num_classes))
        few_shot_query_images = np.zeros(shape=(num_classes * query_sho, self.x_size_image,
                                                self.y_size_image, self.channels))

        # Get a random subset of num_classes labels from the entire label set.

        label_subset = random.sample(self.dirs, k=num_classes)

        num_labels = np.asarray([self.classes_tags.index(elem) for elem in label_subset])

        for class_idx, class_obj in enumerate(label_subset):
            # Use enumerated index value as a temporary label for mini-batch in
            # few shot learning.
            local_folder = self.data_folder + "/" + class_obj

            few_shot_train_labels[class_idx * training_sho: (class_idx + 1) * training_sho] = class_idx
            # If creating a split dataset for testing, select an extra sample from each
            # label to create the test dataset.
            if test_split and not query_split:
                few_shot_test_labels[class_idx * testing_sho: (class_idx + 1) * testing_sho] = class_idx
                # sample random elements to open from the folder
                rand_indexes = random.sample(os.listdir(local_folder), k=training_sho + testing_sho
                                             )
                images_to_split = np.array([np.expand_dims(np.array(plt.imread(local_folder + "/" + index)), -1)
                                            for index in rand_indexes])

                # take just one shot of samples of the k + eval shots taken
                # all the images except from the last one go in training:
                few_shot_train_images[class_idx * training_sho: (class_idx + 1) * training_sho] = \
                    images_to_split[:-testing_sho]

                few_shot_test_images[class_idx * testing_sho: (class_idx + 1) * testing_sho] = \
                    images_to_split[-testing_sho:]  # take last elements

            if query_split and not test_split:
                few_shot_query_labels[class_idx * query_sho: (class_idx + 1) * query_sho] = class_idx

                rand_indexes = random.sample(os.listdir(local_folder), k=training_sho + query_sho)
                images_to_split = np.array([np.expand_dims(np.array(plt.imread(local_folder + "/" + index)), -1)
                                            for index in rand_indexes])

                # take just one shot of samples of the k + eval shots taken
                # all the images except from the last one go in training:
                few_shot_train_images[class_idx * training_sho: (class_idx + 1) * training_sho] \
                    = images_to_split[:-query_sho]

                few_shot_query_images[class_idx * query_sho: (class_idx + 1) * query_sho] = \
                    images_to_split[-query_sho:]  # take last elements

            if query_split and test_split:
                few_shot_test_labels[class_idx * testing_sho: (class_idx + 1) * testing_sho] = class_idx
                few_shot_query_labels[class_idx * query_sho: (class_idx + 1) * query_sho] = class_idx
                # during the evaluation phase I need both query and test split
                rand_indexes = random.sample(os.listdir(local_folder), k=training_sho + testing_sho + query_sho)
                images_to_split = np.array([np.expand_dims(np.array(plt.imread(local_folder + "/" + index)), -1)
                                            for index in rand_indexes])

                # take just one shot of samples of the k + eval shots taken
                # all the images except from the last ones go in training:
                few_shot_train_images[class_idx * training_sho: (class_idx + 1) * training_sho
                ] = images_to_split[:-(query_sho + testing_sho)]

                few_shot_query_images[class_idx * query_sho: (class_idx + 1) * query_sho] = \
                    images_to_split[-(query_sho + testing_sho):-testing_sho]  # take last elements

                few_shot_test_images[class_idx * testing_sho: (class_idx + 1) * testing_sho] = \
                    images_to_split[-testing_sho:]  # take last elements

            if not query_split and not test_split:
                # For each index in the randomly selected label_subset, sample the
                # necessary number of images.
                # without splitting, use all the images for training
                rand_indexes = random.sample(os.listdir(local_folder), k=training_sho)
                trn_images = np.array([np.expand_dims(np.array(plt.imread(local_folder + "/" + index)), -1)
                                       for index in rand_indexes])
                few_shot_train_images[class_idx * training_sho: (class_idx + 1) * training_sho] = trn_images

        if test_split and not query_split:
            return few_shot_train_images, few_shot_train_labels, few_shot_test_images, \
                   few_shot_test_labels, num_labels
        if query_split and not test_split:
            return few_shot_train_images, few_shot_train_labels, few_shot_query_images, \
                   few_shot_query_labels, num_labels
        if test_split and query_split:
            return few_shot_train_images, few_shot_train_labels, few_shot_test_images, \
                   few_shot_test_labels, few_shot_query_images, few_shot_query_labels, num_labels
        else:
            raise NotImplementedError
