import argparse
import json
import os
from pathlib import Path
import re
from utils.box_plot_function import generate_box_plot
from utils.json_functions import read_json
from utils.task_dataset_gen import Dataset
from utils.task_dataset_gen_meta import Dataset_Meta
from algorithms.Weighting_Net import Weighting_Net
from algorithms.MetaWeighting_Net import MetaWeighting_Net
from algorithms.MAML_2nd_Order import Maml2nd_Order
from algorithms.MAML_1st_Order import Maml1st_Order
from utils.text_log_function import generate_text_logs

"""

TF Meta Learning
Author: Gianfranco Mauro
Main script in root folder for running the experiments from terminal

"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="define experimental setup")
    parser.add_argument(
        "--alg",
        help="available meta learning algorithms [weight_net, meta_weight_net, reptile, maml2nd, maml1st, maml_plus]",
        type=str,
        default="maml1st"  # "weight_net"
    )
    parser.add_argument(
        "--n_ways",
        help="number of ways of the experiment [2, 5, 10, ...]",
        type=int,
        default=2  # 5
    )
    parser.add_argument(
        "--n_shots",
        help="number of shots of the experiment [1, 2, 5, 10, ...]",
        type=int,
        default=1
    )
    parser.add_argument(
        "--n_tests",
        help="number of test shots per class [1, 2, 5, 10, ...]",
        type=int,
        default=10
    )
    parser.add_argument(
        "--n_episodes",
        help="number of episodes of the experiment [10, 100, 1000, 22000, ...]",
        type=int,
        default=50  # 22000
    )
    parser.add_argument(
        "--n_query",
        help="number of queries of the experiment [1, 2, 5, 10, ...]",
        type=int,
        default=1
    )
    parser.add_argument(
        "--n_repeats",
        help="number of repetitions of the experiment [1, 2, 5, 10, ...]",
        type=int,
        default=3
    )
    parser.add_argument(
        "--n_box_plots",
        help="number of box plots for the final accuracy plotting [2, 5, 10, ...]",
        type=int,
        default=2,  # 10,
    )
    parser.add_argument(
        "--eval_step",
        help="number of episodes before an evaluation step [1, 2, 5, 10, ...]",
        type=int,
        default=1
    )
    parser.add_argument(
        "--beta1",
        help="beta1 parameter for Adam [0, 0.1, 0.2, 0.5, 1]",
        type=float,
        default=0
    )
    parser.add_argument(
        "--beta2",
        help="beta2 parameter for Adam [0, 0.1, 0.2, 0.5, 1]",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--n_fin_episodes",
        help="number of final test episodes (after generalization learning)[10, 100, 1000, 10000]",
        type=int,
        default=10  # 10000,
    )
    # todo define also loss function and take out beta_1 and 2 as parameters

    args = parser.parse_args()
    alg_name = args.alg
    # inner loop - training on tasks number of shots (samples per class in mini-batch)
    n_shots = int(args.n_shots)
    # number of ways of the experiment
    n_ways = int(args.n_ways)
    # Num of test samples for evaluation per class. Tot Samples = (num of classes * eval_test_shots).
    n_tests = int(args.n_tests)
    # number of inner loop episodes - mini batch tasks training.
    n_episodes = int(args.n_episodes)
    # number of query shots for the algorithm during the training phase
    n_query = int(args.n_query)
    # number of final evaluations of the algorithm
    n_fin_episodes = int(args.n_fin_episodes)
    # number of times that the experiment has to be repeated
    n_repeats = int(args.n_repeats)
    # num of box plots for evaluation
    n_box_plots = int(args.n_box_plots)
    # After how many meta training episodes, make an evaluation?
    eval_inter = int(args.eval_step)
    # adam beta parameters
    beta_1 = float(args.beta1)
    beta_2 = float(args.beta2)

    # number of simulations per boxplot wanted in the final plot
    boxes_eval = int(round(n_episodes / n_box_plots))

    xbox_multiples = []
    xbox_labels = []

    # add all the multiples of boxes eval to a list:
    for count in range(0, n_box_plots):

        if boxes_eval * count <= n_episodes:
            xbox_multiples.append(boxes_eval * count)

    # add the last value if the last multiple is less than It
    if xbox_multiples[-1] < n_episodes:
        xbox_multiples.append(n_episodes)
    # if the number is bigger, than substitute the last value
    elif xbox_multiples[-1] > n_episodes:
        xbox_multiples[-1] = n_episodes

    # # create a list of labels for the x axes in the bar plot

    for counter, multiple in enumerate(xbox_multiples):
        if counter != len(xbox_multiples) - 1:
            # up to the second-last iteration
            xbox_labels.append(str(multiple) + "-" + str(xbox_multiples[counter + 1] - 1))

    print("Box Plots: " + str(xbox_labels))

    # LOAD DATASET CONFIGURATIONS

    dataset_config_file = "configurations/general_config/dataset_config.json"
    dataset_config_file = Path(dataset_config_file)

    dataset_config = read_json(dataset_config_file)

    if alg_name == "weight_net":
        train_dataset = Dataset(training=True, config=dataset_config, classes=n_ways)
        test_dataset = Dataset(training=False, config=dataset_config, classes=n_ways)
    else:
        train_dataset = Dataset_Meta(training=True, config=dataset_config, classes=n_ways)
        test_dataset = Dataset_Meta(training=False, config=dataset_config, classes=n_ways)

    # LOAD PLOT CONFIGURATIONS:
    plot_config_file = "configurations/general_config/plot_config.json"
    plot_config_file = Path(plot_config_file)

    plot_config = read_json(plot_config_file)

    ### INIT ALGORITHM
    if alg_name == "weight_net":  # todo add condition for injection or embedding
        algorithm = Weighting_Net(n_shots, n_ways, n_episodes, n_query, n_tests, train_dataset, test_dataset,
                                  n_repeats, n_box_plots, eval_inter, beta_1, beta_2, xbox_multiples)
    elif alg_name == "meta_weight_net":
        algorithm = MetaWeighting_Net(n_shots, n_ways, n_episodes, n_query, n_tests, train_dataset, test_dataset,
                                      n_repeats, n_box_plots, eval_inter, beta_1, beta_2, xbox_multiples)
    elif alg_name == "maml2nd":
        algorithm = Maml2nd_Order(n_shots, n_ways, n_episodes, n_query, n_tests, train_dataset, test_dataset,
                                  n_repeats, n_box_plots, eval_inter, beta_1, beta_2, xbox_multiples)
    elif alg_name == "maml1st":
        algorithm = Maml1st_Order(n_shots, n_ways, n_episodes, n_query, n_tests, train_dataset, test_dataset,
                                  n_repeats, n_box_plots, eval_inter, beta_1, beta_2, xbox_multiples)
    else:
        algorithm = []
        input()

    # add general variables to a main configuration for the log files
    main_config = algorithm.spec_config.copy()

    main_config["n_shots"] = n_shots
    main_config["n_ways"] = n_ways
    main_config["n_episodes"] = n_episodes
    main_config["n_query"] = n_query
    main_config["n_tests"] = n_tests
    main_config["n_fin_episodes"] = n_fin_episodes
    # TRAIN MODEL FOR N REPETITIONS:
    for repeat in range(n_repeats):

        base_model, training_val_acc, eval_val_acc = algorithm.train_and_evaluate()

        # SAVE MODEL AND LOG FILES
        # create sim_directories
        # assume that no other equal simulations exist

        if not os.path.exists("/results/"):
            os.mkdir("/results/")

        new_directory = "results/" + alg_name + "_" + str(n_shots) + "_Shots_" + \
                        str(n_episodes) + "_Episodes_" + str(repeat) + "_simul_num"

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
            new_directory + "/full_model_weights_" + alg_name + "_" + str(n_episodes) + ".h5")

        #######################
        main_config_file_name = new_directory + "/" + "config.json"

        # SAVE CONFIGURATION FILES:

        with open(main_config_file_name, 'w') as f:
            json.dump(main_config, f, indent=4)

        ## GENERATE BOX PLOTS FOR TRAINING AND EVALUATION

        train_eval_boxes, test_eval_boxes = generate_box_plot(plot_config, alg_name, n_ways, n_episodes,
                                                              new_directory, xbox_labels,
                                                              training_val_acc, eval_val_acc)

        ###################### SAVE BOX PLOTS LOGS ###################

        generate_text_logs(alg_name, new_directory, xbox_labels, n_episodes, train_eval_boxes, test_eval_boxes,
                           training_val_acc, eval_val_acc)

        # FINAL TRAINING/TESTING ON EPISODES AFTER GENERALIZATION TRAINING
        total_accuracy, h, ms_latency, ms_pred_latency = algorithm.final_evaluation(base_model, n_fin_episodes)

        final_accuracy_filename = alg_name + str(n_fin_episodes) + " Test_Tasks_Final_Accuracy.txt"

        final_accuracy_string = "The average accuracy on: " + str(
            n_fin_episodes) + " Test Tasks, with: " + str(
            n_tests) + "samples per class, is: " + str(
            total_accuracy) + "% with a 95% confidence interval of +- " + str(h * 100) + "%"

        final_accuracy_string += "\n" + "The latency time for a final training is: " \
                                 + str(ms_latency) + " milliseconds"

        final_accuracy_string += "\n" + "The latency time for a single prediction is: " \
                                 + str(ms_pred_latency) + " milliseconds"

        with open(new_directory + "/" + final_accuracy_filename, "w") as text_file:
            text_file.write(final_accuracy_string)

    # todo: merge the two task dataset into 1
    # todo: check for names for injection and embedding
    # todo: define parser for injection/embedding and config file
    # todo: complete modification of other algorithms
    # todo: maybe use other script for the final episodes functions
    # todo: save into a log losses results after experiments
    # todo: define a parent class for algorithms