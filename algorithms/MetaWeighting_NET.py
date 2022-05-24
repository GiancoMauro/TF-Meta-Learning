import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import re
import tensorflow as tf
from tensorflow import keras
import time
import warnings
from networks.weighting_modules import Full_Pipeline
from utils.box_plot_function import generate_box_plot
from utils.json_functions import read_json
from utils.statistics import mean_confidence_interval, add_noise_images
from utils.task_dataset_gen_meta import Dataset
from utils.text_log_function import generate_text_logs

main_config_file = "../configurations/main_config.json"
main_config_file = Path(main_config_file)

main_config = read_json(main_config_file)

beta1 = main_config["beta1"]
beta2 = main_config["beta2"]

# number of inner loop iterations - mini batch tasks training. In the paper 200K
meta_iters = main_config["meta_iters"]

# After how many meta training episodes, make an evaluation?
eval_interval = main_config["eval_interval"]

# number of times that the experiment has to be repeated
experiments_num = main_config["num_exp_repetitions"]

# num of box plots for evaluation
num_box_plots = main_config["num_boxplots"]
# number of simulations per boxplot wanted in the final plot
boxes_eval = int(round(meta_iters / num_box_plots))

# inner loop - training on tasks number of shots (samples per class in mini-batch)
support_train_shots = main_config["support_train_shots"]  # inner_shots
# number of query shots for MAML algorithm during the training phase
query_shots = main_config["query_shots"]  # IMPORTANT: USED ALSO IN EVALUATION PHASE

# Num of test samples for evaluation per class. Tot Samples = (num of classes * eval_test_shots). Default 1
test_shots = main_config["test_shots"]
# change to smooth as much as possible the bar-range plot (e.g. if 10 means 10 * num_classes)
classes = main_config["classes"]

# number of final evaluations of the algorithm
number_of_evaluations = main_config["number_of_evaluations_final"]

dataset_config_file = "../configurations/dataset_config.json"
dataset_config_file = Path(dataset_config_file)

dataset_config = read_json(dataset_config_file)

spec_config_file = "../configurations/MetaWeighting_NET.json"
spec_config_file = Path(spec_config_file)

spec_config = read_json(spec_config_file)

Algorithm_name = spec_config["algorithm_name"] + "_"

# feature dimension for the Embedding Module of Weighting Nets.
feature_dimension = spec_config["feature_dimension"]
weighting_dim = spec_config["weighting_dimension"]

# inner loop learning rate of the Adam optimizer:
internal_learning_rate = spec_config["internal_learning_rate"]

plot_config_file = "../configurations/plot_config.json"
plot_config_file = Path(plot_config_file)

plot_config = read_json(plot_config_file)

# Load font sizes:
sup_title_font_size = plot_config["sup_title_font_size"]
title_font_size = plot_config["title_font_size"]
labels_font_size = plot_config["labels_font_size"]
text_font_size = plot_config["text_font_size"]
starting_font_size = plot_config["starting_font_size"]

### Meta related parameters ##
meta_batches = spec_config["meta_batches"]
outer_learning_rate = spec_config["outer_learning_rate"]
base_epochs = spec_config["base_train_epochs"]
# the number of samples per batch is shots * num classes. e.g. 5 shot per class and 4 classes: batch_size = 20
batch_size = classes * support_train_shots

if (classes * support_train_shots) % batch_size == 0:
    # even batch size
    num_batches_per_inner_base_epoch = (classes * support_train_shots) / batch_size
else:
    num_batches_per_inner_base_epoch = round((classes * support_train_shots) / batch_size) + 1

# total number of batches for every meta iteration: needed for BNRS + BNWB
tot_num_base_batches = base_epochs * num_batches_per_inner_base_epoch

Algorithm_name += str(classes) + "_Classes_"

# MSRA initalization
initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                    distribution='truncated_normal', seed=None)

xbox_multiples = []
xbox_labels = []

# add all the multiples of boxes eval to a list:
for count in range(0, num_box_plots):

    if boxes_eval * count <= meta_iters:
        xbox_multiples.append(boxes_eval * count)

# add the last value if the last multiple is less than It
if xbox_multiples[-1] < meta_iters:
    xbox_multiples.append(meta_iters)
# if the number is bigger, than substitute the last value
elif xbox_multiples[-1] > meta_iters:
    xbox_multiples[-1] = meta_iters

# # create a list of labels for the x axes in the bar plot

for counter, multiple in enumerate(xbox_multiples):
    if counter != len(xbox_multiples) - 1:
        # up to the second-last iteration
        xbox_labels.append(str(multiple) + "-" + str(xbox_multiples[counter + 1] - 1))

print("Box Plots: " + str(xbox_labels))

# Load font sizes:
plt.rcParams.update({'font.size': starting_font_size})

train_dataset = Dataset(training=True, config=dataset_config, classes=classes)
test_dataset = Dataset(training=False, config=dataset_config, classes=classes)

for experim_num in range(experiments_num):

    full_pipeline_model = Full_Pipeline(classes, feature_dimension, weighting_dim)

    inner_optimizer = keras.optimizers.Adam(learning_rate=internal_learning_rate, beta_1=beta1, beta_2=beta2)

    outer_optimizer = keras.optimizers.Adam(learning_rate=outer_learning_rate, beta_1=beta1, beta_2=beta2)

    ############## WEIGHTING NET IMPLEMENTATION LOOP ##########################à

    x_eval = []
    general_training_val_acc = []
    general_eval_val_acc = []
    training_val_acc = []
    eval_val_acc = []
    buffer_training_val_acc = []
    buffer_eval_val_acc = []
    mapped_train_eval_labels = []
    mapped_train_eval_predictions = []
    mapped_test_eval_labels = []
    mapped_test_eval_predictions = []

    # Step 2: instead of checking for convergence, we train for a number
    # of epochs

    # Step 3 and 4
    # query_loss_sum is the summation of losses over time for the size of meta_batches
    query_loss_sum = tf.zeros(classes * query_shots)
    query_loss_partial_sum = tf.zeros(classes * query_shots)
    for meta_iter in range(0, meta_iters):
        print(meta_iter)
        # # set the new learning step for the meta optimizer
        # the dataset to contains support and query
        mini_support_dataset, support_images, support_labels, query_images, query_labels, tsk_labels = \
            train_dataset.get_mini_dataset(batch_size, base_epochs, support_train_shots, classes,
                                           query_split=True, query_sho=query_shots)

        if meta_iter == 0:
            full_pipeline_model.call(support_images, query_images, support_train_shots, query_shots)

        old_vars = full_pipeline_model.get_weights()

        epochs_counter = 0
        inner_batch_counter = 0
        # embedded_support = embedding_model(support_images)
        for images, labels in mini_support_dataset:
            num_train_shots_epoch = len(images)
            with tf.GradientTape() as test_tape:

                noise_images = add_noise_images(images)

                # Step 5
                with tf.GradientTape() as train_tape:

                    relat_pred = full_pipeline_model(support_images, noise_images, support_train_shots, 1,
                                                     batch_s=num_train_shots_epoch)
                    # learn the mapping
                    train_loss = keras.losses.sparse_categorical_crossentropy(labels, relat_pred)

                gradients = train_tape.gradient(train_loss, full_pipeline_model.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
                inner_optimizer.apply_gradients(zip(gradients, full_pipeline_model.trainable_weights))

                if epochs_counter == base_epochs - 1:
                    # If I'm at last epoch iteration (done at the end of the inner loop only)
                    # Step 8
                    # compute the model loss over different images (query data)
                    # evaluate model trained on theta' over the query images
                    relat_preds = full_pipeline_model(support_images, query_images, support_train_shots, query_shots)
                    query_loss = keras.losses.sparse_categorical_crossentropy(query_labels, relat_preds)
                    # sum the meta loss for the outer learning every inner batch of the last epoch
                    query_loss_partial_sum = query_loss_partial_sum + query_loss

                    if inner_batch_counter == num_batches_per_inner_base_epoch - 1:
                        # if i'm in the last inner batch, then average the query_loss_sum over the performed batches
                        query_loss_sum += query_loss_partial_sum / num_batches_per_inner_base_epoch
                        # reset the query loss partial sum over inner meta_batches
                        query_loss_partial_sum = tf.zeros(classes * query_shots)

                        if (meta_iter != 0 and meta_iter % meta_batches == 0) or meta_iter == meta_iters - 1:
                            # if I'm at the last meta iter of my batch of meta-tasks
                            # divide the sum of query losses by the number of meta batches
                            # IMPORTANT: by graphs properties in Tensorflow, this operation
                            # has to be done in the gradient loop,
                            # or will set the gradients to zero.
                            # so at the last epoch of the last training epoch before the query update:
                            query_loss_sum = query_loss_sum / meta_batches

            inner_batch_counter += 1
            if inner_batch_counter == num_batches_per_inner_base_epoch:
                inner_batch_counter = 0
                epochs_counter += 1

        # go back on theta parameters for the update
        # META UPDATE IS DONE EVERY DEFINED NUM OF META BATCHES
        full_pipeline_model.set_weights(old_vars)
        # Step 8
        # is it keeping track of the losses on time?
        if (meta_iter != 0 and meta_iter % meta_batches == 0) or meta_iter == meta_iters - 1:
            # Perform optimization for the meta step. Use the final weights obtained after N defined Meta batches

            out_gradients = test_tape.gradient(query_loss_sum, full_pipeline_model.trainable_variables)
            outer_optimizer.apply_gradients(zip(out_gradients, full_pipeline_model.trainable_variables))
            # empty the query_loss_sum for a new cbatch
            query_loss_sum = tf.zeros(classes * query_shots)

        # Evaluation loop
        if meta_iter % eval_interval == 0:
            # if (meta_iter % boxes_eval == 0 or meta_iter == meta_iters - 1) and meta_iter != 0:
            if (meta_iter in xbox_multiples or meta_iter == meta_iters - 1) and meta_iter != 0:
                # when I have enough samples for a box of the box-range plot, add these values to the general list
                # condition: the meta iter is a multiple of the x_box multiples or last iteration and meta_iter is
                # not 0.
                general_training_val_acc.append(buffer_training_val_acc)
                general_eval_val_acc.append(buffer_eval_val_acc)
                buffer_training_val_acc = []
                buffer_eval_val_acc = []

            accuracies = []
            task_specific_labels = []
            task_confusion_matrices = []
            mapped_task_test_labels = []
            mapped_task_test_predictions = []
            for dataset in (train_dataset, test_dataset):
                # set it to zero for validation and then test
                num_correct = 0
                # Sample a mini dataset from the full dataset.
                mini_train_dataset, train_images, train_labels, test_images, test_labels, task_labels = \
                    dataset.get_mini_dataset(batch_size, base_epochs,
                                             support_train_shots, classes, test_split=True,
                                             testing_sho=test_shots)

                eval_preds = full_pipeline_model(train_images, test_images, support_train_shots, test_shots)

                predicted_classes_eval = []
                for prediction_sample in eval_preds:
                    predicted_classes_eval.append(tf.argmax(np.asarray(prediction_sample)))
                for index, prediction in enumerate(predicted_classes_eval):
                    if prediction == test_labels[index]:
                        num_correct += 1

                # after the last epoch, compute the confusion matrix:
                # compute confusion matrix
                conf_matrix = tf.math.confusion_matrix(labels=test_labels,
                                                       predictions=np.asarray(eval_preds).argmax(1)).numpy()

                task_confusion_matrices.append(conf_matrix)
                numeric_task_labels = np.array(task_labels).astype(int)
                task_specific_labels.append(numeric_task_labels)
                # for both validation and testing, accuracy is done over the length of test samples
                accuracies.append(num_correct / len(test_labels))

            # attach the new labels to the main label list
            mapped_train_eval_labels.append(mapped_task_test_labels[:len(test_labels)])
            mapped_train_eval_predictions.append(mapped_task_test_predictions[:len(test_labels)])
            mapped_test_eval_labels.append(mapped_task_test_labels[len(test_labels):])
            mapped_test_eval_predictions.append(mapped_task_test_predictions[len(test_labels):])

            x_eval.append(meta_iter)
            # meta learning test after validation => Validation because it's done on training images not used in the
            # train
            training_val_acc.append(accuracies[0])
            buffer_training_val_acc.append(accuracies[0])
            # test accuracy
            eval_val_acc.append(accuracies[1])
            buffer_eval_val_acc.append(accuracies[1])

            if meta_iter % 5 == 0:  # or meta_iter % meta_batches == 0:
                print("batch %d: eval on train=%f eval on test=%f" % (meta_iter, accuracies[0], accuracies[1]))

    # assume that no other equal simulations exist
    simul_repeat_dir = 1

    new_directory = Algorithm_name + str(meta_iters) + "_shots_" + str(support_train_shots) + "_simul_num_" + \
                    str(experim_num)

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

    full_pipeline_model.save_weights(new_directory + "/full_model_weights_" + Algorithm_name + str(meta_iters) + ".h5")

    #######################
    main_config_file_name = new_directory + "/" + "main_config.json"
    alg_config_file_name = new_directory + "/" + Algorithm_name + str(meta_iters) + "_config.json"

    # SAVE CONFIGURATION FILES:
    with open(alg_config_file_name, 'w') as f:
        json.dump(main_config, f, indent=4)

    with open(main_config_file_name, 'w') as f:
        json.dump(spec_config, f, indent=4)

    ## GENERATE BOX PLOTS FOR TRAINING AND EVALUATION

    train_eval_boxes, test_eval_boxes = generate_box_plot(plot_config, Algorithm_name,  classes, meta_iters, new_directory, xbox_labels,
                                                          general_training_val_acc, general_eval_val_acc)

    ###################### SAVE BOX PLOTS LOGS ###################

    generate_text_logs(Algorithm_name, new_directory, xbox_labels, meta_iters, train_eval_boxes, test_eval_boxes,
                       general_training_val_acc, general_eval_val_acc)

    ############ EVALUATION OVER FINAL TASKS ###############

    # Evaluate the model over the defined number of tasks:

    test_accuracy = []

    time_stamps_single_pred = []

    for task_num in range(0, number_of_evaluations):

        print("final task num: " + str(task_num))
        mini_tr_fin_dataset, train_images_task, train_labels_task, test_images_task, test_labels_task, task_labs = \
            test_dataset.get_mini_dataset(batch_size, base_epochs, support_train_shots, classes, test_split=True,
                                          testing_sho=test_shots)

        # predictions for the task
        eval_preds = full_pipeline_model(train_images_task, test_images_task, support_train_shots, test_shots)

        single_pred_start = time.time()
        pred_example = np.expand_dims(test_images_task[0], 0)
        single_pred = full_pipeline_model(train_images_task, pred_example, support_train_shots, 1, multi_query=False)
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

    total_accuracy = np.average(test_accuracy)

    test_accuracy, h = mean_confidence_interval(np.array(test_accuracy) / 100)

    ms_pred_latency = np.mean(time_stamps_single_pred) * 1e3

    final_accuracy_filename = Algorithm_name + str(number_of_evaluations) + " Test Tasks_FINAL_ACCURACY.txt"

    final_accuracy_string = "The average accuracy on: " + str(number_of_evaluations) + " Test Tasks, with: " + str(
        test_shots) \
                            + "samples per class, is: " + str(
        total_accuracy) + "% with a 95% confidence interval of +- " + str(h * 100) + "%"

    final_accuracy_string += "\n" + "The latency time for a single prediction is: " \
                             + str(ms_pred_latency) + " milliseconds"

    with open(new_directory + "/" + final_accuracy_filename, "w") as text_file:
        text_file.write(final_accuracy_string)
