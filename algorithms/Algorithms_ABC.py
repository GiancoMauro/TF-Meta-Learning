from abc import ABC
import time

class AlgorithmsABC(ABC):
    """
        Core Implementation of the Algorithms
        """

    def __init__(self, alg, n_shots, n_ways, n_episodes, n_query, n_tests, train_dataset, test_dataset,
                 n_repeats, n_box_plots, eval_step, beta1, beta2, xbox_multiples, n_fin_episodes):
        # print(kwargs)
        # self.__dict__.update(kwargs)
        self.alg = alg
        self.beta1 = beta1
        self.beta2 = beta2
        self.episodes = n_episodes
        self.eval_interval = eval_step
        self.experiments_num = n_repeats
        self.num_box_plots = n_box_plots
        self.xbox_multiples = xbox_multiples
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.final_episodes = n_fin_episodes

        self.n_ways = n_ways
        self.support_train_shots = n_shots
        self.query_shots = n_query

        self.test_shots = n_tests

    # def final_evaluation(self, base_model, final_episodes):
    #     """
    #     Function that computes the performances of the generated generalization models of a set of final tasks
    #     :param base_model: generalization model
    #     :param final_episodes: number of final episodes for the evaluation
    #     :return: total_accuracy, h, ms_prediction_latency
    #     """
    #     ############ EVALUATION OVER FINAL TASKS ###############
    #     test_accuracy = []
    #
    #     base_weights = base_model.get_weights()
    #
    #     time_stamps_adaptation = []
    #     time_stamps_single_predict = []
    #
    #     inner_optimizer = keras.optimizers.Adam(learning_rate=self.internal_learning_rate,
    #                                             beta_1=self.beta1, beta_2=self.beta2)
    #
    #     for task_num in range(0, final_episodes):
    #
    #         print("final task num: " + str(task_num))
    #         train_images_task, train_labels_task, test_images_task, test_labels_task, task_labs = \
    #             self.test_dataset.get_mini_dataset(self.support_train_shots, self.n_ways,
    #                                                test_split=True, testing_sho=self.test_shots)
    #
    #         # generate tf dataset:
    #         train_set_task = tf.data.Dataset.from_tensor_slices(
    #             (train_images_task.astype(np.float32), train_labels_task.astype(np.int32))
    #         )
    #         train_set_task = train_set_task.shuffle(100).batch(self.batch_size).repeat(self.eval_train_epochs)
    #
    #         # train the Base model over the 1-shot Task:
    #         adaptation_start = time.time()
    #         for images, labels in train_set_task:
    #             with tf.GradientTape() as tape:
    #                 preds = base_model(images)
    #                 loss = keras.losses.sparse_categorical_crossentropy(labels, preds)
    #             grads = tape.gradient(loss, base_model.trainable_weights)
    #             inner_optimizer.apply_gradients(zip(grads, base_model.trainable_weights))
    #         adaptation_end = time.time()
    #         time_stamps_adaptation.append(adaptation_end - adaptation_start)
    #         # predictions for the task
    #         eval_predicts = base_model.predict(test_images_task)
    #
    #         single_prediction_start = time.time()
    #         prediction_example = np.expand_dims(test_images_task[0], 0)
    #         base_model.predict(prediction_example)
    #         single_prediction_end = time.time()
    #         time_stamps_single_predict.append(single_prediction_end - single_prediction_start)
    #
    #         predicted_classes = []
    #         for prediction_sample in eval_predicts:
    #             predicted_classes.append(tf.argmax(np.asarray(prediction_sample)))
    #
    #         num_correct_out_loop = 0
    #         for index, prediction in enumerate(predicted_classes):
    #             if prediction == test_labels_task[index]:
    #                 num_correct_out_loop += 1
    #
    #         test_accuracy_new_val = num_correct_out_loop / len(test_images_task)
    #         test_accuracy.append(round(test_accuracy_new_val * 100, 2))
    #
    #         # reset the network weights to the base ones
    #         # Reset the weights after getting the evaluation accuracies.
    #         base_model.set_weights(base_weights)
    #
    #     total_accuracy = np.average(test_accuracy)
    #
    #     test_accuracy, h = mean_confidence_interval(np.array(test_accuracy) / 100)
    #
    #     ms_latency = np.mean(time_stamps_adaptation) * 1e3
    #
    #     ms_prediction_latency = np.mean(time_stamps_single_predict) * 1e3
    #
    #     return total_accuracy, h, ms_latency, ms_prediction_latency