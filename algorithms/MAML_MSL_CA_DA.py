"""
Author: Gianfranco Mauro
Partial Tensorflow Implementation of the algorithm: "How to train your MAML".
Antoniou, Antreas, Harrison Edwards, and Amos Storkey.
"How to train your MAML." arXiv preprint arXiv:1810.09502 (2018).

The following contributes from the paper have been implemented in this tensorflow version of MAML:
1.  Multi-Step Loss Optimization (MSL)
2.  Cosine Annealing of Meta-Optimizer Learning Rate (CA)
3.  Derivative-Order Annealing (DA)

https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from algorithms.Algorithms_abc import AlgorithmsABC
from networks.conv_modules import conv_base_model
from utils.json_functions import read_json
from utils.statistics import mean_confidence_interval


class Mamlplus(AlgorithmsABC):
    """
        Core Implementation of the Maml+MSL+CA+DA algorithm
        """

    def __init__(self, **kwargs):
        super(Mamlplus, self).__init__(**kwargs)

        self.alg_name = "Maml+MSL+CA+DA_"

        spec_config_file = "configurations/MAML+MSL+CA+DA.json"
        spec_config_file = Path(spec_config_file)

        self.spec_config = read_json(spec_config_file)

        # inner loop learning rate of the Adam optimizer:
        self.internal_learning_rate = self.spec_config["internal_learning_rate"]

        # ### META LEARNING ADAPTIVE RATE INSTEAD OF OUTER ADAM ##
        self.initial_outer_learning_rate = self.spec_config["initial_outer_learning_rate"]

        # size of the training and evaluation batches (independent by the number of sample per class)
        self.batch_size = self.spec_config["batch_size"]

        # how many training repetitions over the mini-batch in task learning phase?
        self.base_train_epochs = self.spec_config["base_train_epochs"]  # (inner_iters per new tasks)

        # how many training repetitions over the mini-batch in evaluation phase?  EVAL TASK
        self.eval_train_epochs = self.spec_config["eval_train_epochs"]

        # for CA
        self.decay_steps = self.spec_config["decay_steps"]
        # base weights for MSL
        self.loss_weights = self.spec_config["initial_loss_weights"]

        self.conv_filters_number = self.spec_config["conv_filters_per_layer"]
        self.conv_kernel_size = self.spec_config["conv_kernel_size"]

        if (self.n_ways * self.support_train_shots) % self.batch_size == 0:
            # even batch size
            self.num_batches_per_inner_base_epoch = (self.n_ways * self.support_train_shots) / self.batch_size
        else:
            self.num_batches_per_inner_base_epoch = round(
                (self.n_ways * self.support_train_shots) / self.batch_size) + 1

        # total number of batches for every meta iteration: needed for BNRS + BNWB
        self.tot_num_base_batches = self.base_train_epochs * self.num_batches_per_inner_base_epoch

        self.alg_name += str(self.n_ways) + "_Classes_"

    def train_and_evaluate(self):
        """
        main function for the training and evaluation of the meta learning algorithm

        :return: base_model, general_training_val_acc, general_eval_val_acc
        """
        base_model = conv_base_model(self.n_ways, self.conv_filters_number, self.conv_kernel_size)
        base_model.compile()
        base_model.summary()
        inner_optimizer = keras.optimizers.Adam(learning_rate=self.internal_learning_rate, beta_1=self.beta1,
                                                beta_2=self.beta2)

        outer_learning_rate_schedule = tf.keras.experimental.CosineDecay(self.initial_outer_learning_rate,
                                                                         self.decay_steps)

        outer_optimizer = keras.optimizers.Adam(learning_rate=outer_learning_rate_schedule,
                                                beta_1=self.beta1, beta_2=self.beta2)

        ############### MAML IMPLEMENTATION LOOP ##########################à
        general_training_val_acc = []
        general_eval_val_acc = []
        training_val_acc = []
        eval_val_acc = []
        buffer_training_val_acc = []
        buffer_eval_val_acc = []
        # Step 2: instead of checking for convergence, we train for a number
        # of epochs

        # Step 3 and 4
        # query_loss_sum is the summation of losses over time for the size of meta_batches
        query_loss_sum = tf.zeros(self.n_ways * self.query_shots)
        # query loss partial sum is the average of the loss over the inner meta_batches
        query_loss_partial_sum = tf.zeros(self.n_ways * self.query_shots)
        for episode in range(0, self.episodes):
            print(episode)
            # the dataset to contains support and query
            train_images, train_labels, query_images, query_labels, tsk_labels = \
                self.train_dataset.get_mini_dataset(self.support_train_shots, self.n_ways, query_split=True,
                                                    query_sho=self.query_shots
                                                    )

            # generate tf dataset:
            mini_support_dataset = tf.data.Dataset.from_tensor_slices(
                (train_images.astype(np.float32), train_labels.astype(np.int32))
            )
            mini_support_dataset = mini_support_dataset.shuffle(100).batch(self.batch_size).repeat(
                self.base_train_epochs)

            # MSL ANNEALING IN META EPOCHS:
            if episode == round(self.episodes / 4):
                self.loss_weights = [0.01, 0.03, 0.10, 0.88]
            if episode == round(self.episodes / 2):
                self.loss_weights = [0.001, 0.004, 0.045, 0.95]
            if episode == round(3 * self.episodes / 4):
                self.loss_weights = [0, 0.002, 0.008, 0.99]
            # weights are normalized to Sum: 1
            epochs_counter = 0
            inner_batches_counter = 0
            # IMPORTANT, in MAML, the external update depends by the weighted sum of the loss over epochs
            old_vars = base_model.get_weights()
            for images, labels in mini_support_dataset:
                if episode <= 50:  # (1/4) * episodes:
                    # print("FIRST ORDER")
                    # 1st ORDER MAML

                    # Step 5
                    with tf.GradientTape() as train_tape:
                        support_preds = base_model(images)
                        train_loss = keras.losses.sparse_categorical_crossentropy(labels, support_preds)
                    # Step 6

                    gradients = train_tape.gradient(train_loss, base_model.trainable_variables)
                    inner_optimizer.apply_gradients(zip(gradients, base_model.trainable_weights))
                    with tf.GradientTape() as test_tape:
                        # Step 8
                        # compute the model loss over different images (query data)
                        # evaluate model trained on theta' over the query images
                        query_preds = base_model(query_images)
                        query_loss = keras.losses.sparse_categorical_crossentropy(query_labels, query_preds)
                        # sum the meta loss for the outer learning every N defined Meta Batches
                        query_loss_partial_sum = query_loss_partial_sum + query_loss * self.loss_weights[epochs_counter]

                        if inner_batches_counter == self.num_batches_per_inner_base_epoch - 1:
                            # if i'm in the last inner batch, then average the query_loss_sum over the performed batches
                            query_loss_sum += query_loss_partial_sum / self.num_batches_per_inner_base_epoch
                            # reset the query loss partial sum over inner meta_batches
                            query_loss_partial_sum = tf.zeros(self.n_ways * self.query_shots)
                else:
                    # print("SECOND ORDER")
                    # 2nd Order MAML

                    with tf.GradientTape() as test_tape:
                        # Step 5
                        with tf.GradientTape() as train_tape:
                            support_preds = base_model(images)
                            train_loss = keras.losses.sparse_categorical_crossentropy(labels, support_preds)

                        # Step 6
                        gradients = train_tape.gradient(train_loss, base_model.trainable_variables)
                        # Internal Lr = External
                        inner_optimizer.apply_gradients(zip(gradients, base_model.trainable_weights))
                        # for all the weights: *weights = weights - learning rate*(Delta update)

                        # Step 8
                        # compute the model loss over different images (query data)
                        # evaluate model trained on theta' over the query images
                        query_preds = base_model(query_images)
                        query_loss = keras.losses.sparse_categorical_crossentropy(query_labels, query_preds)
                        # sum the meta loss for the outer learning every N defined epochs
                        query_loss_partial_sum = query_loss_partial_sum + query_loss * self.loss_weights[epochs_counter]
                        # since the sum of the loss weights is one, no division of the query loss over inner epochs is 
                        # needed 

                        if inner_batches_counter == self.num_batches_per_inner_base_epoch - 1:
                            # if i'm in the last inner batch, then average the query_loss_sum over the performed batches
                            query_loss_sum += query_loss_partial_sum / self.num_batches_per_inner_base_epoch
                            # reset the query loss partial sum over inner inner_batches
                            query_loss_partial_sum = tf.zeros(self.n_ways * self.query_shots)

                inner_batches_counter += 1

                if inner_batches_counter == self.num_batches_per_inner_base_epoch:
                    # update the current epochs weight
                    epochs_counter += 1
                    inner_batches_counter = 0

            # go back on theta parameters for the update after THE INNER LOOP
            # META UPDATE IS DONE EVERY DEFINED NUM OF INNER EPOCHS
            base_model.set_weights(old_vars)

            # UPDATE USING THE WEIGHTED SUM OVER INNER LOOPS
            gradients = test_tape.gradient(query_loss_sum, base_model.trainable_variables)
            outer_optimizer.apply_gradients(zip(gradients, base_model.trainable_variables))
            # empty the query_loss_sum for a new batch
            query_loss_sum = tf.zeros(self.n_ways * self.query_shots)
            # Step 8
            # is it keeping track of the losses on time?

            # Evaluation loop
            if episode % self.eval_interval == 0:
                if (episode in self.xbox_multiples or episode == self.episodes - 1) and episode != 0:
                    # when I have enough samples for a box of the box-range plot, add these values to the general
                    # list condition: the meta iter is a multiple of the x_box multiples or last iteration and
                    # episode is not 0.
                    general_training_val_acc.append(buffer_training_val_acc)
                    general_eval_val_acc.append(buffer_eval_val_acc)
                    # print("before:" + str(general_training_val_acc))
                    buffer_training_val_acc = []
                    buffer_eval_val_acc = []
                    # print("after:" + str(general_training_val_acc))

                accuracies = []
                for dataset in (self.train_dataset, self.test_dataset):
                    # set it to zero for validation and then test
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
                            preds = base_model(images)
                            loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
                        grads = tape.gradient(loss, base_model.trainable_weights)
                        inner_optimizer.apply_gradients(zip(grads, base_model.trainable_weights))

                    # test phase after model evaluation
                    eval_preds = base_model.predict(test_images)
                    predicted_classes_eval = []
                    for prediction_sample in eval_preds:
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
        test_accuracy = []

        base_weights = base_model.get_weights()

        time_stamps_adaptation = []
        time_stamps_single_pred = []

        inner_optimizer = keras.optimizers.Adam(learning_rate=self.internal_learning_rate,
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
                    preds = base_model(images)
                    loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
                grads = tape.gradient(loss, base_model.trainable_weights)
                inner_optimizer.apply_gradients(zip(grads, base_model.trainable_weights))
            adaptation_end = time.time()
            time_stamps_adaptation.append(adaptation_end - adaptation_start)
            # predictions for the task
            eval_preds = base_model.predict(test_images_task)

            single_pred_start = time.time()
            pred_example = np.expand_dims(test_images_task[0], 0)
            single_pred = base_model.predict(pred_example)
            single_pred_end = time.time()
            time_stamps_single_pred.append(single_pred_end - single_pred_start)

            predicted_classes = []
            for prediction_sample in eval_preds:
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

        ms_prediction_latency = np.mean(time_stamps_single_pred) * 1e3

        return total_accuracy, h, ms_latency, ms_prediction_latency
