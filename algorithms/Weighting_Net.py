"""
Author: Gianfranco Mauro
Tensorflow Implementation of the algorithm:
"Injection-Weighting-Net".
Mauro, Martinez-Rodriguez, Ott, Servadei, Wille, Cuellar and Morales-Santos.
"Context-Adaptable Radar-Based People Counting via Few-Shot Learning."
Springer Applied Intelligence (2023).

Base Implementation from pytorch versions:

"User-definable Dynamic Hand Gesture Recognition Based on Doppler Radar and Few-shot Learning"
Zeng, Xianglong, Chaoyang Wu, and Wen-Bin Ye.
"User-Definable Dynamic Hand Gesture Recognition Based on Doppler Radar and Few-Shot Learning."
IEEE Sensors Journal 21.20 (2021): 23224-23233.

https://github.com/AGroupofProbiotocs/WeighingNet

and:

"Learning to Compare: Relation Network for Few-Shot Learning"
Sung, Flood, et al. "Learning to compare: Relation network for few-shot learning."
Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

https://github.com/floodsung/LearningToCompare_FSL
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from algorithms.Algorithms_ABC import AlgorithmsABC
from networks.weighting_modules import Full_Pipeline
from utils.json_functions import read_json
from utils.statistics import mean_confidence_interval


class Weighting_Net(AlgorithmsABC):
    """
    Core Implementation of the Weighting Network algorithm (Both Embedding and Injection)
    """

    def __init__(self, **kwargs):
        super(Weighting_Net, self).__init__(**kwargs)

        self.alg_name = "Weighting_Net_"

        spec_config_file = "configurations/Weighting_Net.json"
        spec_config_file = Path(spec_config_file)

        self.spec_config = read_json(spec_config_file)

        # embedding dimension for the Embedding/Injection Module of Weighting Nets.
        self.embedding_dimension = self.spec_config["embedding_dimension"]
        self.weighting_dim = self.spec_config["weighting_dimension"]

        # is a simulation with injection module?
        self.is_injection = self.spec_config["injection_module"]

        # inner loop learning rate of the Adam optimizer:
        self.internal_learning_rate = self.spec_config["internal_learning_rate"]

        # # MSRA initialization
        # initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
        #                                                     distribution='truncated_normal', seed=None)

    def train_and_evaluate(self):
        """
        main function for the training and evaluation of the meta learning algorithm

        :return: full_pipeline_model, general_training_val_acc, general_eval_val_acc
        """
        if self.is_injection:
            self.alg_name += "Injection_"
        else:
            self.alg_name += "Embedding_"

        self.alg_name += str(self.n_ways) + "_Classes_"

        full_pipeline_model = Full_Pipeline(self.n_ways, self.embedding_dimension, self.weighting_dim,
                                            is_injection=self.is_injection)

        inner_optimizer = tf.keras.optimizers.Adam(learning_rate=self.internal_learning_rate, beta_1=self.beta1,
                                                beta_2=self.beta2)

        ############## WEIGHTING NET IMPLEMENTATION LOOP ##########################à

        general_training_val_acc = []
        general_eval_val_acc = []
        training_val_acc = []
        eval_val_acc = []
        buffer_training_val_acc = []
        buffer_eval_val_acc = []

        # Step 2: instead of checking for convergence, we train for a number
        # of epochs

        # Step 3 and 4
        for episode in range(0, self.episodes):
            print(episode)
            # # set the new learning step for the meta optimizer
            # the dataset to contains support and query
            support_images, support_labels, query_images, query_labels, tsk_labels = \
                self.train_dataset.get_mini_dataset(self.support_train_shots, self.n_ways,
                                                    query_split=True, query_sho=self.query_shots)

            if episode == 0:
                full_pipeline_model.call(support_images, query_images, self.support_train_shots, self.query_shots)

            with tf.GradientTape() as train_tape:

                relational_predict = full_pipeline_model(support_images, query_images, self.support_train_shots,
                                                         self.query_shots)

                # learn the mapping
                train_loss = tf.keras.losses.sparse_categorical_crossentropy(query_labels, relational_predict)

            gradients = train_tape.gradient(train_loss, full_pipeline_model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            inner_optimizer.apply_gradients(zip(gradients, full_pipeline_model.trainable_weights))

            # Step 8
            # is it keeping track of the losses on time?

            # Evaluation loop
            if episode % self.eval_interval == 0:
                if (episode in self.xbox_multiples or episode == self.episodes - 1) and episode != 0:
                    # when I have enough samples for a box of the box-range plot, add these values to the general list
                    # condition: the episode is a multiple of the x_box multiples or last iteration and episode is
                    # not 0.
                    general_training_val_acc.append(buffer_training_val_acc)
                    general_eval_val_acc.append(buffer_eval_val_acc)
                    buffer_training_val_acc = []
                    buffer_eval_val_acc = []

                accuracies = []
                for dataset in (self.train_dataset, self.test_dataset):
                    # set it to zero for validation and then test
                    num_correct = 0
                    # Sample a mini dataset from the full dataset.
                    train_images, train_labels, test_images, test_labels, task_labels = dataset.get_mini_dataset(
                        self.support_train_shots, self.n_ways, test_split=True,
                        testing_sho=self.test_shots)

                    eval_predicts = full_pipeline_model(train_images, test_images, self.support_train_shots,
                                                        self.test_shots)

                    predicted_classes_eval = []
                    for prediction_sample in eval_predicts:
                        predicted_classes_eval.append(tf.argmax(np.asarray(prediction_sample)))
                    for index, prediction in enumerate(predicted_classes_eval):
                        if prediction == test_labels[index]:
                            num_correct += 1

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
                    print("episode %d: eval on train=%f eval on test=%f" % (episode, accuracies[0], accuracies[1]))

        return full_pipeline_model, general_training_val_acc, general_eval_val_acc

    def final_evaluation(self, full_pipeline_model, final_episodes):
        """
        Function that computes the performances of the generated generalization models of a set of final tasks
        :param full_pipeline_model: generalization model
        :param final_episodes: number of final episodes for the evaluation
        :return: total_accuracy, h, ms_prediction_latency
        """
        ############ EVALUATION OVER FINAL TASKS ###############

        # Evaluate the model over the defined number of tasks:

        test_accuracy = []

        time_stamps_single_predict = []

        for task_num in range(0, final_episodes):

            print("final task num: " + str(task_num))
            train_images_task, train_labels_task, test_images_task, test_labels_task, _ = \
                self.test_dataset.get_mini_dataset(self.support_train_shots, self.n_ways, test_split=True,
                                                   testing_sho=self.test_shots)

            # predictions for the task
            eval_predicts_fin = full_pipeline_model(train_images_task, test_images_task,
                                                self.support_train_shots, self.test_shots)

            single_prediction_start = time.time()
            prediction_example = np.expand_dims(test_images_task[0], 0)
            full_pipeline_model(train_images_task, prediction_example, self.support_train_shots, 1,
                                multi_query=False)
            single_prediction_end = time.time()
            time_stamps_single_predict.append(single_prediction_end - single_prediction_start)

            predicted_classes = []
            for prediction_sample in eval_predicts_fin:
                predicted_classes.append(tf.argmax(np.asarray(prediction_sample)))

            num_correct_out_loop = 0
            for index, prediction in enumerate(predicted_classes):
                if prediction == test_labels_task[index]:
                    num_correct_out_loop += 1

            test_accuracy_new_val = num_correct_out_loop / len(test_images_task)
            test_accuracy.append(round(test_accuracy_new_val * 100, 2))

        total_accuracy = np.average(test_accuracy)

        test_accuracy, h = mean_confidence_interval(np.array(test_accuracy) / 100)

        # No adaptation training required with relational algorithms
        ms_latency = 0

        ms_prediction_latency = np.mean(time_stamps_single_predict) * 1e3

        return total_accuracy, h, ms_latency, ms_prediction_latency
