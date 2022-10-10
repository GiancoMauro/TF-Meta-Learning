"""
Author: Gianfranco Mauro
Partial testing re-implementation of the Algorithm:
"On First-Order Meta-Learning Algorithms".
Nichol, Alex, Joshua Achiam, and John Schulman.
"On first-order meta-learning algorithms." arXiv preprint arXiv:1803.02999 (2018).

Original Implementation from Keras: ADMoreau: Few-Shot learning with Reptile
https://keras.io/examples/vision/reptile/
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from algorithms.Algorithms_ABC import AlgorithmsABC
from networks.conv_modules import Conv_Dense_Module
from utils.json_functions import read_json
from utils.statistics import mean_confidence_interval


class Reptile(AlgorithmsABC):
    """
        Core Implementation of the Reptile algorithm
        """

    def __init__(self, **kwargs):  # pass local variables
        super(Reptile, self).__init__(**kwargs)

        self.alg_name = "Reptile_"

        spec_config_file = "configurations/Reptile.json"
        spec_config_file = Path(spec_config_file)

        self.spec_config = read_json(spec_config_file)

        self.conv_filters_number = self.spec_config["conv_filters_per_layer"]
        self.conv_kernel_size = self.spec_config["conv_kernel_size"]

        # inner loop learning rate of the Adam optimizer:
        self.internal_learning_rate = self.spec_config["internal_learning_rate"]

        # step size for weights update over the mini-batch iterations. Bigger it is, bigger are the meta step updates...
        self.meta_step_size = self.spec_config["meta_step_size"]
        # outer step size -- learning importance over meta steps

        # size of the training and evaluation batches (independent by the number of sample per class)
        self.batch_size = self.spec_config["batch_size"]

        # how many training repetitions over the mini-batch in task learning phase?
        self.base_train_epochs = self.spec_config["base_train_epochs"]  # (inner_iters per new tasks)

        # how many training repetitions over the mini-batch in evaluation phase?  EVAL TASK
        self.eval_train_epochs = self.spec_config["eval_train_epochs"]

        if (self.n_ways * self.support_train_shots) % self.batch_size == 0:
            # even batch size
            self.num_batches_per_inner_base_epoch = (self.n_ways * self.support_train_shots) / self.batch_size
        else:
            self.num_batches_per_inner_base_epoch = round(
                (self.n_ways * self.support_train_shots) / self.batch_size) + 1

        self.alg_name += str(self.n_ways) + "_Classes_"

    def train_and_evaluate(self):
        """
        main function for the training and evaluation of the meta learning algorithm

        :return: base_model, general_training_val_acc, general_eval_val_acc
        """

        base_model = Conv_Dense_Module(self.n_ways, self.conv_filters_number, self.conv_kernel_size)
        optimizer = keras.optimizers.Adam(learning_rate=self.internal_learning_rate,
                                          beta_1=self.beta1, beta_2=self.beta2)

        general_training_val_acc = []
        general_eval_val_acc = []
        training_val_acc = []
        eval_val_acc = []
        buffer_training_val_acc = []
        buffer_eval_val_acc = []
        ##### episodes loop #####

        for episode in range(self.episodes):
            print(episode)
            frac_done = episode / self.episodes
            cur_meta_step_size = (1 - frac_done) * self.meta_step_size
            # Temporarily save the weights from the model.
            old_vars = base_model.get_weights()
            # Get a sample from the full dataset.
            train_images, train_labels, _, _, tsk_labels = \
                self.train_dataset.get_mini_dataset(self.support_train_shots, self.n_ways, query_split=True,
                                                    query_sho=self.query_shots
                                                    )
            if episode == 0:
                # init model
                base_model.call(train_images)
                base_model.build(train_images.shape)
                base_model.summary()

            # generate tf dataset:
            mini_support_dataset = tf.data.Dataset.from_tensor_slices(
                (train_images.astype(np.float32), train_labels.astype(np.int32))
            )
            mini_support_dataset = mini_support_dataset.shuffle(100).batch(self.batch_size).repeat(
                self.eval_train_epochs)

            for images, labels in mini_support_dataset:
                # for each data batch
                with tf.GradientTape() as tape:
                    # random initialization of weights
                    predicts = base_model(images)
                    loss = keras.losses.sparse_categorical_crossentropy(labels, predicts)
                grads = tape.gradient(loss, base_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, base_model.trainable_weights))
            new_vars = base_model.get_weights()
            # Perform optimization for the meta step. Use the final weights obtained after N inner batches
            for var in range(len(new_vars)):
                new_vars[var] = old_vars[var] + (
                        (new_vars[var] - old_vars[var]) * cur_meta_step_size
                )
            # After the meta-learning step, reload the newly-trained weights into the model.
            base_model.set_weights(new_vars)
            # Evaluation loop
            if episode % self.eval_interval == 0:
                if (episode in self.xbox_multiples or episode == self.episodes - 1) and episode != 0:
                    # when I have enough samples for a box of the box-range plot, add these values to the general list
                    # condition: the meta iter is a multiple of the x_box multiples or last iteration and meta_iter is
                    # not 0.
                    general_training_val_acc.append(buffer_training_val_acc)
                    general_eval_val_acc.append(buffer_eval_val_acc)
                    buffer_training_val_acc = []
                    buffer_eval_val_acc = []

                accuracies = []
                for dataset in (self.train_dataset, self.test_dataset):
                    # set it to zero for validation and then test)
                    num_correct = 0
                    # Sample a mini dataset from the full dataset.
                    train_images_eval, train_labels_eval, test_images, test_labels, task_labels = \
                        dataset.get_mini_dataset(self.support_train_shots, self.n_ways, test_split=True,
                                                 testing_sho=self.test_shots)

                    # generate tf dataset:
                    train_set = tf.data.Dataset.from_tensor_slices(
                        (train_images_eval.astype(np.float32), train_labels_eval.astype(np.int32))
                    )
                    train_set = train_set.shuffle(100).batch(self.batch_size).repeat(self.eval_train_epochs)

                    old_vars = base_model.get_weights()
                    # Train on the samples and get the resulting accuracies.
                    for images, labels in train_set:
                        with tf.GradientTape() as tape:
                            predicts = base_model(images)
                            loss = keras.losses.sparse_categorical_crossentropy(labels, predicts)
                        grads = tape.gradient(loss, base_model.trainable_weights)
                        optimizer.apply_gradients(zip(grads, base_model.trainable_weights))

                    # test phase after model evaluation
                    eval_predicts = base_model.predict(test_images)
                    predicted_classes_eval = []
                    for prediction_sample in eval_predicts:
                        predicted_classes_eval.append(tf.argmax(np.asarray(prediction_sample)))
                    for index, prediction in enumerate(predicted_classes_eval):
                        if prediction == test_labels[index]:
                            num_correct += 1

                    # Reset the weights after getting the evaluation accuracies.
                    base_model.set_weights(old_vars)
                    # for both validation and testing, accuracy is done over the length of test samples
                    accuracies.append(num_correct / len(test_labels))

                # meta learning test after validation => Validation because it's done on training images not used in the
                # train
                training_val_acc.append(accuracies[0])
                buffer_training_val_acc.append(accuracies[0])
                # test accuracy
                eval_val_acc.append(accuracies[1])
                buffer_eval_val_acc.append(accuracies[1])

                if episode % 5 == 0:
                    print("batch %d: eval on train=%f eval on test=%f" % (episode, accuracies[0], accuracies[1]))

        return base_model, general_training_val_acc, general_eval_val_acc

    def final_evaluation(self, base_model, final_episodes):
        """
        Function that computes the performances of the generated generalization models of a set of final tasks
        :param base_model: generalization model
        :param final_episodes: number of final episodes for the evaluation
        :return: total_accuracy, h, ms_prediction_latency
        """
        ############ EVALUATION OVER FINAL TASKS ###############

        ############ EVALUATION OVER FINAL TASKS ###############

        test_accuracy = []

        base_weights = base_model.get_weights()

        time_stamps_adaptation = []
        time_stamps_single_predict = []

        optimizer = keras.optimizers.Adam(learning_rate=self.internal_learning_rate,
                                          beta_1=self.beta1, beta_2=self.beta2)

        for task_num in range(0, final_episodes):

            print("final task num: " + str(task_num))
            train_images_task, train_labels_task, test_images_task, test_labels_task, task_labs = \
                self.test_dataset.get_mini_dataset(self.support_train_shots, self.n_ways,
                                                   test_split=True, testing_sho=self.test_shots)

            # generate tf dataset:
            train_set_task = tf.data.Dataset.from_tensor_slices(
                (train_images_task.astype(np.float32), train_labels_task.astype(np.int32))
            )
            train_set_task = train_set_task.shuffle(100).batch(self.batch_size).repeat(self.eval_train_epochs)

            # train the Base model over the 1-shot Task:
            adaptation_start = time.time()
            for images, labels in train_set_task:
                with tf.GradientTape() as tape:
                    predicts = base_model(images)
                    loss = keras.losses.sparse_categorical_crossentropy(labels, predicts)
                grads = tape.gradient(loss, base_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, base_model.trainable_weights))
            adaptation_end = time.time()
            time_stamps_adaptation.append(adaptation_end - adaptation_start)
            # predictions for the task
            eval_predicts = base_model.predict(test_images_task)

            single_predict_start = time.time()
            predict_example = np.expand_dims(test_images_task[0], 0)
            base_model.predict(predict_example)
            single_predict_end = time.time()
            time_stamps_single_predict.append(single_predict_end - single_predict_start)

            predicted_classes = []
            for prediction_sample in eval_predicts:
                predicted_classes.append(tf.argmax(np.asarray(prediction_sample)))

            num_correct_out_loop = 0
            for index, prediction in enumerate(predicted_classes):
                if prediction == test_labels_task[index]:
                    num_correct_out_loop += 1

            test_accuracy_new_val = num_correct_out_loop / len(test_images_task)
            test_accuracy.append(round(test_accuracy_new_val * 100, 2))

            # reset the network weights to the base ones
            # Reset the weights after getting the evaluation accuracies.
            base_model.set_weights(base_weights)

        total_accuracy = np.average(test_accuracy)

        test_accuracy, h = mean_confidence_interval(np.array(test_accuracy) / 100)

        ms_latency = np.mean(time_stamps_adaptation) * 1e3

        ms_prediction_latency = np.mean(time_stamps_single_predict) * 1e3

        return total_accuracy, h, ms_latency, ms_prediction_latency
