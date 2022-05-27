"""
Author: Gianfranco Mauro
1st Order Tensorflow Implementation of the algorithm:
"Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks".
Finn, Chelsea, Pieter Abbeel, and Sergey Levine. "Model-agnostic meta-learning for fast adaptation of deep networks."
International conference on machine learning. PMLR, 2017."

https://github.com/cbfinn/maml
"""

import json
import numpy as np
import os
from pathlib import Path
import re
import tensorflow as tf
from tensorflow import keras
import time
import warnings
from networks.conv_modules import conv_base_model
from utils.box_plot_function import generate_box_plot
from utils.json_functions import read_json
from utils.statistics import mean_confidence_interval
from utils.task_dataset_gen_meta import Dataset
from utils.text_log_function import generate_text_logs

main_config_file = "../configurations/main_config.json"
main_config_file = Path(main_config_file)

main_config = read_json(main_config_file)

beta1 = main_config["beta1"]
beta2 = main_config["beta2"]

# number of inner loop episodes - mini batch tasks training. In the paper 200K
episodes = main_config["episodes"]

# After how many meta training episodes, make an evaluation?
eval_interval = main_config["eval_interval"]

# number of times that the experiment has to be repeated
experiments_num = main_config["num_exp_repetitions"]

# num of box plots for evaluation
num_box_plots = main_config["num_boxplots"]
# number of simulations per boxplot wanted in the final plot
boxes_eval = int(round(episodes / num_box_plots))

# inner loop - training on tasks number of shots (samples per class in mini-batch)
support_train_shots = main_config["support_train_shots"]  # inner_shots
# number of query shots for MAML algorithm during the training phase
query_shots = main_config["query_shots"]  # IMPORTANT: USED ALSO IN EVALUATION PHASE

# Num of test samples for evaluation per class. Tot Samples = (num of classes * eval_test_shots). Default 1
test_shots = main_config["test_shots"]
# images used to test the algorithm after the few shot learning phase.
# outer loop - final training on tasks number of shots. Num of training samples = (num of classes * final_train_shots)
# Num of test samples for final testing per class. Tot Samples = (num of classes * test_shots)
# test_shots = 200
# 80 * num classes (5) = 400

# change to smooth as much as possible the bar-range plot (e.g. if 10 means 10 * num_classes)
classes = main_config["classes"]

# number of final evaluations of the algorithm
number_of_evaluations = main_config["number_of_evaluations_final"]

dataset_config_file = "../configurations/general_config/dataset_config.json"
dataset_config_file = Path(dataset_config_file)

dataset_config = read_json(dataset_config_file)

spec_config_file = "../configurations/MAML.json"
spec_config_file = Path(spec_config_file)

spec_config = read_json(spec_config_file)

Algorithm_name = spec_config["algorithm_name"] + "_"

# inner loop learning rate of the Adam optimizer:
internal_learning_rate = spec_config["internal_learning_rate"]

# outer loop learning rate of the Adam optimizer:
outer_learning_rate = spec_config["outer_learning_rate"]

# size of the training and evaluation batches (independent by the number of sample per class)
batch_size = spec_config["batch_size"]

# how many training repetitions over the mini-batch in task learning phase?
base_train_epochs = spec_config["base_train_epochs"]  # (inner_iters per new tasks)

# how many training repetitions over the mini-batch in evaluation phase?  EVAL TASK
eval_train_epochs = spec_config["eval_train_epochs"]

# number of meta batches for the generalization update
meta_batches = spec_config["meta_batches"]

plot_config_file = "../configurations/general_config/plot_config.json"
plot_config_file = Path(plot_config_file)

plot_config = read_json(plot_config_file)

if (classes * support_train_shots) % batch_size == 0:
    # even batch size
    num_batches_per_inner_base_epoch = (classes * support_train_shots) / batch_size
else:
    num_batches_per_inner_base_epoch = round((classes * support_train_shots) / batch_size) + 1

# total number of batches for every meta iteration: needed for BNRS + BNWB
tot_num_base_batches = base_train_epochs * num_batches_per_inner_base_epoch

xbox_multiples = []
xbox_labels = []

# add all the multiples of boxes eval to a list:
for count in range(0, num_box_plots):

    if boxes_eval * count <= episodes:
        xbox_multiples.append(boxes_eval * count)

# add the last value if the last multiple is less than It
if xbox_multiples[-1] < episodes:
    xbox_multiples.append(episodes)
# if the number is bigger, than substitute the last value
elif xbox_multiples[-1] > episodes:
    xbox_multiples[-1] = episodes

# # create a list of labels for the x axes in the bar plot

for counter, multiple in enumerate(xbox_multiples):
    if counter != len(xbox_multiples) - 1:
        # up to the second-last iteration
        xbox_labels.append(str(multiple) + "-" + str(xbox_multiples[counter + 1] - 1))

print("Box Plots: " + str(xbox_labels))

train_dataset = Dataset(training=True, config=dataset_config, classes=classes)
test_dataset = Dataset(training=False, config=dataset_config, classes=classes)

for experim_num in range(experiments_num):

    base_model = conv_base_model(dataset_config, spec_config, classes)
    base_model.compile()
    base_model.summary()
    optimizer = keras.optimizers.Adam(learning_rate=internal_learning_rate, beta_1=beta1, beta_2=beta2)

    inner_optimizer = keras.optimizers.Adam(learning_rate=internal_learning_rate, beta_1=beta1, beta_2=beta2)

    outer_optimizer = keras.optimizers.Adam(learning_rate=outer_learning_rate, beta_1=beta1, beta_2=beta2)

    ############### MAML IMPLEMENTATION LOOP ##########################à
    
    general_training_val_acc = []
    general_eval_val_acc = []
    training_val_acc = []
    eval_val_acc = []
    buffer_training_val_acc = []
    buffer_eval_val_acc = []
    
    # Step 2: instead of checking for convergence, we train for a number
    # of epochs
    total_loss = 0
    losses = []
    
    # Step 3 and 4
    # query_loss_sum is the summation of losses over time for the size of meta_batches
    query_loss_sum = tf.zeros(classes * query_shots)
    # query loss partial sum is the average of the loss over the inner meta_batches
    query_loss_partial_sum = tf.zeros(classes*query_shots)
    for episode in range(0, episodes):
        print(episode)
        # # set the new learning step for the meta optimizer
        # the dataset to contains support and query
        mini_support_dataset, _, _, query_images, query_labels, tsk_labels = train_dataset.get_mini_dataset(
            batch_size, base_train_epochs, support_train_shots, classes, query_split=True,
            query_sho=query_shots
        )
        old_vars = base_model.get_weights()
    
        # INTERNAL TRAINING LOOP
        epochs_counter = 0
        inner_batch_counter = 0
        for images, labels in mini_support_dataset:
            # Step 5
            with tf.GradientTape() as train_tape:
                support_preds = base_model(images)
                train_loss = keras.losses.sparse_categorical_crossentropy(labels, support_preds)
            # Step 6
    
            gradients = train_tape.gradient(train_loss, base_model.trainable_variables)
            inner_optimizer.apply_gradients(zip(gradients, base_model.trainable_weights))
    
            # OUTER META LOOP, use the learnt weights in the internal loop to evaluate the query images
            if epochs_counter == base_train_epochs - 1:
                # if I'm in the last epoch, I do the query evaluation
                with tf.GradientTape() as test_tape:
                    # Step 8
                    # compute the model loss over different images (query data)
                    # evaluate model trained on theta' over the query images
                    query_preds = base_model(query_images)
                    query_loss = keras.losses.sparse_categorical_crossentropy(query_labels, query_preds)
                    # sum the meta loss for the outer learning every inner batch of the last epoch
                    query_loss_partial_sum = query_loss_partial_sum + query_loss
    
                    if inner_batch_counter == num_batches_per_inner_base_epoch - 1:
                        # if i'm in the last inner batch, then average the query_loss_sum over the performed batches
                        query_loss_sum += query_loss_partial_sum / num_batches_per_inner_base_epoch
                        # reset the query loss partial sum over inner meta_batches
                        query_loss_partial_sum = tf.zeros(classes * query_shots)
    
                        # sum the meta loss for the outer learning every N defined Meta Batches
                        if (episode != 0 and episode % meta_batches == 0) or episode == episodes - 1:
                            # if I'm at the last meta iter of my batch of meta-tasks
                            # divide the sum of query losses by the number of meta batches
                            # IMPORTANT: by graphs properties in Tensorflow,
                            # this operation has to be done in the gradient loop,
                            # or will set the gradients to zero.
                            # so at the last epoch of the last training epoch before the query update:
                            query_loss_sum = query_loss_sum / meta_batches
            inner_batch_counter += 1
            if inner_batch_counter == num_batches_per_inner_base_epoch:
                inner_batch_counter = 0
                epochs_counter += 1
    
        # go back on theta parameters for the update from the previous METAITER
        base_model.set_weights(old_vars)
    
        if (episode != 0 and episode % meta_batches == 0) or episode == episodes - 1:
            # Perform optimization for the meta step. Use the final weights obtained after N defined Meta batches
            # with first order optimization test tape is not over training tape
            gradients = test_tape.gradient(query_loss_sum, base_model.trainable_variables)
            outer_optimizer.apply_gradients(zip(gradients, base_model.trainable_variables))
            # empty the query_loss_sum for a new cbatch
            query_loss_sum = tf.zeros(classes * query_shots)
    
        # Evaluation loop
        if episode % eval_interval == 0:
            if (episode in xbox_multiples or episode == episodes - 1) and episode != 0:
                # when I have enough samples for a box of the box-range plot, add these values to the general list
                # condition: the meta iter is a multiple of the x_box multiples or last iteration and episode is not 0.
                general_training_val_acc.append(buffer_training_val_acc)
                general_eval_val_acc.append(buffer_eval_val_acc)
                buffer_training_val_acc = []
                buffer_eval_val_acc = []
    
            accuracies = []
            task_specific_tags = []
            mapped_task_test_labels = []
            mapped_task_test_predictions = []
            for dataset in (train_dataset, test_dataset):
                # set it to zero for validation and then test
                num_correct = 0
                # Sample a mini dataset from the full dataset.
                train_set, _, _, test_images, test_labels, task_labels = dataset.get_mini_dataset(
                    batch_size, eval_train_epochs, support_train_shots, classes, test_split=True,
                    testing_sho=test_shots)
                # print(train_set)
    
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

                numeric_task_labels = np.array(task_labels).astype(int)
                task_specific_tags.append(numeric_task_labels)
    
                # transform labels and predictions in tags values
                for lab_index, predict_lab in enumerate(eval_preds.argmax(1)):
                    mapped_task_test_predictions.append(numeric_task_labels[predict_lab])
                    mapped_task_test_labels.append(numeric_task_labels[int(test_labels[lab_index])])
    
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
    
            if episode % 5 == 0:  # or episode % meta_batches == 0:
                print("batch %d: eval on train=%f eval on test=%f" % (episode, accuracies[0], accuracies[1]))

    # create sim_directories
    # assume that no other equal simulations exist
    simul_repeat_dir = 1

    if not os.path.exists("../results/"):
        os.mkdir("../results/")

    Algorithm_name += "1st_Order_"

    Algorithm_name += str(classes) + "_Classes_"

    new_directory = "../results/" + Algorithm_name + str(support_train_shots) + "_Shots_" + \
                    str(episodes) + "_Episodes_" + str(experim_num) + "_simul_num"

    if not os.path.exists(new_directory):
        os.mkdir(new_directory)
    else:
        Pattern = re.compile(new_directory[:-1])
        folder_list = os.listdir()
        filtered = [folder for folder in folder_list if Pattern.match(folder)]
        list_existing_folders = []
        for directory in list(filtered):
            # find the last existing repetition of the simulation
            list_existing_folders.append(int(str(directory)[-1]))

        simul_repeat_dir = max(list_existing_folders) + 1
        new_directory = new_directory[:-len(str(simul_repeat_dir))] + "_" + str(simul_repeat_dir)
        os.mkdir(new_directory)

    # SAVE MODEL:

    base_model.save_weights(
        new_directory + "/base_model_weights_" + Algorithm_name + str(episodes) + ".h5")

    #######################
    main_config_file_name = new_directory + "/" + "main_config.json"
    alg_config_file_name = new_directory + "/" + Algorithm_name + str(episodes) + "_config.json"

    # SAVE CONFIGURATION FILES:
    with open(alg_config_file_name, 'w') as f:
        json.dump(main_config, f, indent=4)

    with open(main_config_file_name, 'w') as f:
        json.dump(spec_config, f, indent=4)

    ## GENERATE BOX PLOTS FOR TRAINING AND EVALUATION

    train_eval_boxes, test_eval_boxes = generate_box_plot(plot_config, Algorithm_name, classes, episodes,
                                                          new_directory, xbox_labels,
                                                          general_training_val_acc, general_eval_val_acc)

    ###################### SAVE BOX PLOTS LOGS ###################

    generate_text_logs(Algorithm_name, new_directory, xbox_labels, episodes, train_eval_boxes, test_eval_boxes,
                       general_training_val_acc, general_eval_val_acc)

    test_accuracy = []

    base_weights = base_model.get_weights()

    time_stamps_adaptation = []
    time_stamps_single_pred = []

    inner_optimizer = keras.optimizers.Adam(learning_rate=internal_learning_rate, beta_1=beta1, beta_2=beta2)

    for task_num in range(0, number_of_evaluations):

        print("final task num: " + str(task_num))
        train_set_task, _, _, test_images_task, test_labels_task, task_labs = test_dataset.get_mini_dataset(
            batch_size, eval_train_epochs, support_train_shots, classes, test_split=True, testing_sho=test_shots)
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

    ms_pred_latency = np.mean(time_stamps_single_pred) * 1e3

    final_accuracy_filename = Algorithm_name + str(number_of_evaluations) + " Test Tasks_FINAL_ACCURACY.txt"

    final_accuracy_string = "The average accuracy on: " + str(number_of_evaluations) + " Test Tasks, with: " + str(
        test_shots) + "samples per class, is: " + str(
        total_accuracy) + "% with a 95% confidence interval of +- " + str(h * 100) + "%"

    final_accuracy_string += "\n" + "The latency time for a final training is: " \
                             + str(ms_latency) + " milliseconds"

    final_accuracy_string += "\n" + "The latency time for a single prediction is: " \
                             + str(ms_pred_latency) + " milliseconds"

    with open(new_directory + "/" + final_accuracy_filename, "w") as text_file:
        text_file.write(final_accuracy_string)
