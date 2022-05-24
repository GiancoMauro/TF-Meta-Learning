import numpy as np


def generate_text_logs(alg_name, new_directory, xbox_labels, meta_iters, train_eval_boxes, test_eval_boxes,
                       general_training_val_acc, general_eval_val_acc):

    """
    functions that produces a logs in regards to the box range plots according to the
    achieved results over meta iterations (both for training and test evaluations)

    :param alg_name: name of the algorithm from the specific .json file
    :param new_directory: directory where the plot has to be saved
    :param xbox_labels: labels of the box plots that will be generated
    :param meta_iters: number of meta_iterations on which the model has been trained
    :param train_eval_boxes: box plots computed over training meta tasks
    :param test_eval_boxes: box plots computed over test meta tasks
    :param general_training_val_acc: list of accuracy values obtained on training validation
    :param general_eval_val_acc: list of accuracy values obtained on testing validation

    :return: None ( Save the logs as .txt files in the provided folder )
    """

    algorithm_name = alg_name.replace("_", " ") + " "

    train_values = algorithm_name + ": \n"

    # Train Plots VALUES:
    train_values += "TRAIN VALUES \n\nMedian Values: \n"

    for plot_counter, medline_train in enumerate(train_eval_boxes["medians"]):
        current_plot = xbox_labels[plot_counter]
        linedata = medline_train.get_ydata()
        median = linedata[0]
        train_values += current_plot + ", median:" + str(round(median, 2) * 100) + "% \n"

    train_values += "\nAverage Values: \n"

    for plot_counter, avgline_train in enumerate(train_eval_boxes["means"]):
        current_plot = xbox_labels[plot_counter]
        linedata = avgline_train.get_ydata()
        mean = linedata[0]

        train_values += current_plot + ", mean:" + str(round(mean, 2) * 100) + "% \n"

    train_values += "\n Quartiles: \n"

    for plot_counter, current_train_list in enumerate(general_training_val_acc):
        current_plot = xbox_labels[plot_counter]
        lower_quartile = np.percentile(np.array(current_train_list), 25)
        upper_quartile = np.percentile(np.array(current_train_list), 75)
        IQR = upper_quartile - lower_quartile
        lower_whisker = lower_quartile - 1.5 * IQR
        upper_whisker = upper_quartile + 1.5 * IQR
        upper_whisker = np.compress(current_train_list <= upper_whisker, current_train_list)
        lower_whisker = np.compress(current_train_list >= lower_whisker, current_train_list)
        upper_whisker = np.max(upper_whisker)
        lower_whisker = np.min(lower_whisker)
        if upper_whisker < upper_quartile:
            upper_whisker = upper_quartile
        if lower_whisker > lower_quartile:
            lower_whisker = lower_quartile
        train_values += current_plot + ", lower_whisker:" + str(round(lower_whisker, 2) * 100) + "% \n"
        train_values += current_plot + ", lower_quartile:" + str(round(lower_quartile, 2) * 100) + "% \n"
        train_values += current_plot + ", upper_quartile:" + str(round(upper_quartile, 2) * 100) + "% \n"
        train_values += current_plot + ", upper_whisker:" + str(round(upper_whisker, 2) * 100) + "% \n"
    #
    # Test Plot VALUES:
    test_values = algorithm_name + ": \n"

    test_values += "TEST VALUES \n\nMedian Values: \n"

    for plot_counter, medline_test in enumerate(test_eval_boxes["medians"]):
        current_plot = xbox_labels[plot_counter]
        linedata = medline_test.get_ydata()
        median = linedata[0]
        test_values += current_plot + ", median:" + str(round(median, 2) * 100) + "% \n"

    test_values += "\nAverage Values: \n"

    for plot_counter, avgline_test in enumerate(test_eval_boxes["means"]):
        current_plot = xbox_labels[plot_counter]
        linedata = avgline_test.get_ydata()
        mean = linedata[0]

        test_values += current_plot + ", mean:" + str(round(mean, 2) * 100) + "% \n"

    test_values += "\n Quartiles & Whiskers: \n"

    for plot_counter, current_test_list in enumerate(general_eval_val_acc):
        current_plot = xbox_labels[plot_counter]
        lower_quartile = np.percentile(np.array(current_test_list), 25)
        upper_quartile = np.percentile(np.array(current_test_list), 75)
        IQR = upper_quartile - lower_quartile
        lower_whisker = lower_quartile - 1.5 * IQR
        upper_whisker = upper_quartile + 1.5 * IQR
        upper_whisker = np.compress(current_test_list <= upper_whisker, current_test_list)
        lower_whisker = np.compress(current_test_list >= lower_whisker, current_test_list)
        upper_whisker = np.max(upper_whisker)
        lower_whisker = np.min(lower_whisker)
        if upper_whisker < upper_quartile:
            upper_whisker = upper_quartile
        if lower_whisker > lower_quartile:
            lower_whisker = lower_quartile
        test_values += current_plot + ", lower_whisker:" + str(round(lower_whisker, 2) * 100) + "% \n"
        test_values += current_plot + ", lower_quartile:" + str(round(lower_quartile, 2) * 100) + "% \n"
        test_values += current_plot + ", upper_quartile:" + str(round(upper_quartile, 2) * 100) + "% \n"
        test_values += current_plot + ", upper_whisker:" + str(round(upper_whisker, 2) * 100) + "% \n"

    train_values_filename = new_directory + "/" + algorithm_name + str(meta_iters) + "_BoxPlot_Train_results.txt"

    test_values_filename = new_directory + "/" + algorithm_name + str(meta_iters) + "_BoxPlot_Test_results.txt"

    with open(train_values_filename, "w") as text_file:
        text_file.write(train_values)

    with open(test_values_filename, "w") as text_file:
        text_file.write(test_values)
