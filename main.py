import argparse
import json
import os
import re
from pathlib import Path

from algorithms.MAML_1st_Order import Maml1st_Order
from algorithms.MAML_2nd_Order import Maml2nd_Order
from algorithms.MAML_MSL_CA_DA import Maml_Plus
from algorithms.MAMW import MAMW
from algorithms.Reptile import Reptile
from algorithms.Weighting_Net import Weighting_Net
from utils.box_plot_function import generate_box_plot
from utils.boxplots_vs_normal_distr_function import generate_boxplot_vs_normal_dist
from utils.json_functions import read_json
from utils.task_dataset_gen import Dataset
from utils.text_log_function import generate_text_logs

"""

TF Meta Learning
Author: Gianfranco Mauro
Main script in root folder for running the experiments from terminal

"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Define Experimental Setup")
    parser.add_argument(
        "--alg",
        help="available meta learning algorithms [weight_net, mamw, reptile, maml2nd, maml1st, maml_plus]",
        type=str,
        default="weight_net"
    )
    parser.add_argument(
        "--n_ways",
        help="number of ways of the experiment [2, 5, 10, ...]",
        type=int,
        default=5
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
        default=20  # 22000
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
        default=1
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
    parser.add_argument(
        "--results_dir",
        help="name of the subdirectory where to save the results",
        type=str,
        default= "results"
    )
    # todo define also loss function and take out beta_1 and 2 as parameters

    parser_args = parser.parse_args()

    args = vars(parser_args)

    alg_name = parser_args.alg
    # inner loop - training on tasks number of shots (samples per class in mini-batch)
    n_shots = int(parser_args.n_shots)
    # number of ways of the experiment
    n_ways = int(parser_args.n_ways)
    # Num of test samples for evaluation per class. Tot Samples = (num of classes * eval_test_shots).
    n_tests = int(parser_args.n_tests)
    # number of inner loop episodes - mini batch tasks training.
    n_episodes = int(parser_args.n_episodes)

    # number of final evaluations of the algorithm
    n_fin_episodes = int(parser_args.n_fin_episodes)
    # number of times that the experiment has to be repeated
    n_repeats = int(parser_args.n_repeats)
    # num of box plots for evaluation
    n_box_plots = int(parser_args.n_box_plots)

    results_dir = str(parser_args.results_dir) + "/"

    print("Currently Running {}, with {} support and {} tests shots. {}-Way.".format(alg_name, n_shots,
                                                                                        n_tests, n_ways))

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

    print("Generated Box Plots: {}".format(str(xbox_labels)))

    # LOAD DATASET CONFIGURATIONS

    dataset_config_file = "configurations/general_config/dataset_config.json"
    dataset_config_file = Path(dataset_config_file)

    dataset_config = read_json(dataset_config_file)

    # initialize classes tags
    # dataset_config["classes_tags"] = []

    train_dataset = Dataset(training=True, config=dataset_config, classes=n_ways)
    test_dataset = Dataset(training=False, config=dataset_config, classes=n_ways)

    # add dataset and box plots multiples to dictionary
    args["train_dataset"] = train_dataset
    args["test_dataset"] = test_dataset
    args["xbox_multiples"] = xbox_multiples

    # LOAD PLOT CONFIGURATIONS:
    plot_config_file = "configurations/general_config/plot_config.json"
    plot_config_file = Path(plot_config_file)

    plot_config = read_json(plot_config_file)

    ### INIT ALGORITHM
    if alg_name == "weight_net":  # todo add condition for injection or embedding
        algorithm = Weighting_Net(**args)
    elif alg_name == "mamw":
        algorithm = MAMW(**args)
    elif alg_name == "maml2nd":
        algorithm = Maml2nd_Order(**args)
    elif alg_name == "maml1st":
        algorithm = Maml1st_Order(**args)

    elif alg_name == "maml_plus":
        algorithm = Maml_Plus(**args)

    elif alg_name == "reptile":
        algorithm = Reptile(**args)
    else:
        raise ValueError("{} not in the list of available algorithms".format(alg_name))

    # add general variables to a main configuration for the log files
    main_config = {**algorithm.spec_config.copy(), **args}
    # TRAIN MODEL FOR N REPETITIONS:
    for repeat in range(n_repeats):

        base_model, training_val_acc, eval_val_acc = algorithm.train_and_evaluate()

        # todo SAVE MODEL AND LOG FILES
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        new_directory = "{}{}_{}_Shots_{}_Ways_{}_Episodes_{}_sim_num".format(results_dir, alg_name, str(n_shots),
                                                                              str(n_ways), str(n_episodes),
                                                                              str(repeat))

        if not os.path.exists(new_directory):
            os.mkdir(new_directory)
        else:
            Pattern = re.compile(new_directory[len(results_dir):])
            folder_list = os.listdir(results_dir)
            filtered = [folder for folder in folder_list if Pattern.match(folder)]
            list_existing_folders = []
            for directory in list(filtered):
                # find the last existing repetition of the simulation
                pattern = r"Episodes_(\d+)_sim"
                match = re.search(pattern, directory)
                list_existing_folders.append(int(match.group(1)))

            simulation_repeat_dir = max(list_existing_folders) + 100
            new_directory = "{}{}_{}_Shots_{}_Ways_{}_Episodes_{}_sim_num".format(results_dir, alg_name, str(n_shots),
                                                                                  str(n_ways), str(n_episodes),
                                                                                  str(simulation_repeat_dir))

            os.mkdir(new_directory)

        # SAVE MODEL:

        weights_save_path =  "{}/full_model_weights_{}_{}.h5".format(new_directory, alg_name, str(n_episodes))

        base_model.save_weights(weights_save_path)

        #######################
        main_config_file_name = new_directory + "/" + "config.json"

        # SAVE CONFIGURATION FILES:

        with open(main_config_file_name, 'w') as f:
            json.dump(main_config, f, indent=4, default=lambda obj: obj.__dict__)

        ## GENERATE BOX PLOTS FOR TRAINING AND EVALUATION

        train_eval_boxes, test_eval_boxes = generate_box_plot(plot_config, alg_name, n_ways, n_episodes,
                                                              new_directory, xbox_labels,
                                                              training_val_acc, eval_val_acc)

        ############ OPTIONAL GENERATION OF BOX PLOTS VS NORMAL DISTRIBUTION ##############
        generate_boxplot_vs_normal_dist(plot_config, alg_name, new_directory, n_episodes, xbox_labels, n_ways,
                                        eval_val_acc)

        ###################### SAVE BOX PLOTS LOGS ###################

        generate_text_logs(alg_name, new_directory, xbox_labels, n_episodes, train_eval_boxes, test_eval_boxes,
                           training_val_acc, eval_val_acc)

        # FINAL TRAINING/TESTING ON EPISODES AFTER GENERALIZATION TRAINING
        total_accuracy, h, ms_latency, ms_predict_latency = algorithm.final_evaluation(base_model, n_fin_episodes)

        final_accuracy_filename = "{}{}  Test_Tasks_Final_Accuracy.txt".format(alg_name, str(n_fin_episodes))

        final_accuracy_string = \
            "The average accuracy on: {} Test Tasks, with: {} samples per class, is: {}% with a " \
            "95% confidence interval of +- {}%".format(str(n_fin_episodes), n_tests, str(total_accuracy),
                                                       str(round(h * 100, 3)))

        final_accuracy_string += "\n The latency time for a final training is: {} milliseconds".format(str(ms_latency))

        final_accuracy_string += "\n The latency time for a single prediction is: " \
                                 "{} milliseconds".format(str(ms_predict_latency))

        with open(new_directory + "/" + final_accuracy_filename, "w") as text_file:
            text_file.write(final_accuracy_string)

    # todo: check for names for injection and embedding
    # todo: define parser for injection/embedding and config file
    # todo: maybe use main Alg script for the final episodes functions
    # todo: save into a log losses results after experiments
