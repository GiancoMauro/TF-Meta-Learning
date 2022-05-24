import matplotlib.pyplot as plt


def generate_box_plot(config, alg_name, classes, meta_iters, new_directory, xbox_labels,
                      general_training_val_acc, general_eval_val_acc):

    """
    Functions that saves as a figure the trend of box range plots over meta iteration time

    :param config: configuration file used for the simulation
    :param alg_name: name of the algorithm from the specific .json file
    :param classes: number of classes of the meta-task
    :param meta_iters: number of meta_iterations on which the model has been trained
    :param new_directory: directory where the plot has to be saved
    :param xbox_labels: labels of the box plots that will be generated
    :param general_training_val_acc: list of accuracy values obtained on training validation
    :param general_eval_val_acc: list of accuracy values obtained on testing validation

    :return: obtained box plots on training and test (train_eval_boxes, test_eval_boxes)
    """

    algorithm_name = alg_name.replace("_", " ") + " "

    sup_title_font_size = config["sup_title_font_size"]
    title_font_size = config["title_font_size"]
    labels_font_size = config["labels_font_size"]

    # plot boxplot
    fig1, axs = plt.subplots(2)
    fig1.set_figheight(17)
    fig1.set_figwidth(25)
    fig1_title = algorithm_name + " - Box Plot"

    fig1.suptitle(fig1_title, fontsize=sup_title_font_size)
    fig1.subplots_adjust(hspace=.25)

    train_eval_boxes = axs[0].boxplot(general_training_val_acc, patch_artist=True, vert=True, showmeans=True)
    test_eval_boxes = axs[1].boxplot(general_eval_val_acc, patch_artist=True, vert=True, showmeans=True)

    axs[0].set_title("Evaluation on Training Set", fontsize=title_font_size)
    axs[1].set_title("Evaluation on Test Set", fontsize=title_font_size)
    axs[0].set_xlabel("Box Plots - Set Of Meta Iterations", fontsize=labels_font_size)
    axs[0].set_ylabel("Accuracy over " + str(classes) + " classes", fontsize=labels_font_size)
    axs[1].set_xlabel("Box Plots - Set Of Meta Iterations", fontsize=labels_font_size)
    axs[1].set_ylabel("Accuracy over " + str(classes) + " classes", fontsize=labels_font_size)

    for ax in axs:
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.yaxis.grid(True, linestyle='dashed', color="#C0C0C0")
        ax.set_xticklabels(xbox_labels)

    # change colors and style of the boxplots

    ## EVAL TRAIN STYLES
    for box in train_eval_boxes['boxes']:
        # change outline color
        box.set(color='#8C1E0A', linewidth=2)
        # change fill color
        box.set_facecolor('#E13F22')
    for whisker in train_eval_boxes['whiskers']:
        whisker.set(color='#4A0F04', linestyle='-', linewidth=2)
    for cap in train_eval_boxes['caps']:
        cap.set(color='#4A0F04', linewidth=2)
    for median in train_eval_boxes['medians']:
        median.set(color='#E7E751', linewidth=2)
    ## change the style of fliers and their fill
    for flier in train_eval_boxes['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.9)

    # EVAL TEST STYLES
    for box in test_eval_boxes['boxes']:
        # change outline color
        box.set(color='#0A3D8C', linewidth=2)
        # change fill color
        box.set_facecolor('#2276E1')
    for whisker in test_eval_boxes['whiskers']:
        whisker.set(color='#04084A', linestyle='-', linewidth=2)
    for cap in test_eval_boxes['caps']:
        cap.set(color='#04084A', linewidth=2)
    for median in test_eval_boxes['medians']:
        median.set(color='#8ADFD2', linewidth=2)
    ## change the style of fliers and their fill
    for flier in test_eval_boxes['fliers']:
        flier.set(marker='o', color='#2991E7', alpha=0.9)

    # FACE COLORS
    face_colors = ['#8C1E0A', '#0A3D8C']

    for patch in train_eval_boxes['boxes']:
        patch.set_facecolor(face_colors[0])

    for patch in test_eval_boxes['boxes']:
        patch.set_facecolor(face_colors[1])

    plt.savefig(new_directory + "/" + algorithm_name + str(meta_iters) + "_iterations_box_range.pdf", dpi=400)

    return train_eval_boxes, test_eval_boxes