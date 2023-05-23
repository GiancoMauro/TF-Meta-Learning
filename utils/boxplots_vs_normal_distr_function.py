import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def generate_boxplot_vs_normal_dist(config, alg_name, new_directory, episodes, xbox_labels, classes,
                                    general_eval_val_acc):
    """

    :param config: configuration file used for the simulation
    :param alg_name: name of the algorithm from the specific .json file
    :param new_directory: directory where the plot has to be saved
    :param episodes: number of episodes on which the model has been trained
    :param xbox_labels: labels of the box plots that will be generated
    :param classes: number of classes of the meta-task
    :param general_eval_val_acc: list of accuracy values obtained on testing validation

    :return: None ( Save the obtained figure in the provided folder )
    """

    algorithm_name = alg_name.replace("_", " ") + " "

    sup_title_font_size = config["sup_title_font_size"]
    title_font_size = config["title_font_size"]
    labels_font_size = config["labels_font_size"]
    text_font_size = config["text_font_size"]

    fig3, axs_new = plt.subplots(nrows=3, ncols=2, sharex="row")
    fig3.set_figheight(24)
    fig3.set_figwidth(27)
    fig3_title = algorithm_name + " - First Vs Last Test Box Plot: Accuracy Distribution"

    fig3.suptitle(fig3_title, fontsize=sup_title_font_size)
    fig3.subplots_adjust(hspace=.25)
    first_eval_boxes = axs_new[0, 0].boxplot(general_eval_val_acc[0], patch_artist=True, showmeans=True, vert=False)
    last_eval_boxes = axs_new[0, 1].boxplot(general_eval_val_acc[-1], patch_artist=True, showmeans=True, vert=False)

    axs_new[0, 0].get_xaxis().tick_bottom()
    axs_new[0, 0].get_yaxis().tick_left()
    axs_new[0, 0].xaxis.grid(True, linestyle='dashed', color="#D3D3D3")
    axs_new[0, 0].set_yticklabels([xbox_labels[0]])

    axs_new[0, 1].get_xaxis().tick_bottom()
    axs_new[0, 1].get_yaxis().tick_left()
    axs_new[0, 1].xaxis.grid(True, linestyle='dashed', color="#D3D3D3")
    axs_new[0, 1].set_yticklabels([xbox_labels[-1]])
    x_label_string = "Accuracy over " + str(classes) + " classes"

    axs_new[0, 0].set_title("First Episodes Test Box Plot", fontsize=title_font_size)
    axs_new[0, 1].set_title("Last Episodes Test Box Plot", fontsize=title_font_size)
    axs_new[0, 0].set_ylabel("Episodes", fontsize=labels_font_size)
    axs_new[0, 0].set_xlabel(x_label_string, fontsize=labels_font_size)
    axs_new[0, 1].set_ylabel("Episodes", fontsize=labels_font_size)
    axs_new[0, 1].set_xlabel(x_label_string, fontsize=labels_font_size)
    axs_new[1, 0].set_xlabel(x_label_string, fontsize=labels_font_size)
    axs_new[1, 1].set_xlabel(x_label_string, fontsize=labels_font_size)

    axs_new[1, 0].set_title("Probability Distribution Function - First", fontsize=title_font_size)
    axs_new[1, 1].set_title("Probability Distribution Function - Last", fontsize=title_font_size)

    axs_new[2, 0].set_title("Density Histogram - First", fontsize=title_font_size)
    axs_new[2, 1].set_title("Density Histogram - Last", fontsize=title_font_size)

    axs_new[2, 0].set_ylabel('probability density', fontsize=labels_font_size)
    axs_new[2, 1].set_ylabel('probability density', fontsize=labels_font_size)
    axs_new[2, 0].set_xlabel(x_label_string, fontsize=labels_font_size)
    axs_new[2, 1].set_xlabel(x_label_string, fontsize=labels_font_size)

    # FIRST EVAL BOX
    for box in first_eval_boxes['boxes']:
        # change outline color
        box.set(color='#0A3D8C', linewidth=2)
        # change fill color
        box.set_facecolor('#2276E1')
    for whisker in first_eval_boxes['whiskers']:
        whisker.set(color='#04084A', linestyle='-', linewidth=2)
    for cap in first_eval_boxes['caps']:
        cap.set(color='#04084A', linewidth=2)
    for median in first_eval_boxes['medians']:
        median.set(color='#8ADFD2', linewidth=2)
    ## change the style of fliers and their fill
    for flier in first_eval_boxes['fliers']:
        flier.set(marker='o', color='#2991E7', alpha=0.9)

    # LAST EVAL BOX
    for box in last_eval_boxes['boxes']:
        # change outline color
        box.set(color='#0A3D8C', linewidth=2)
        # change fill color
        box.set_facecolor('#2276E1')
    for whisker in last_eval_boxes['whiskers']:
        whisker.set(color='#04084A', linestyle='-', linewidth=2)
    for cap in last_eval_boxes['caps']:
        cap.set(color='#04084A', linewidth=2)
    for median in last_eval_boxes['medians']:
        median.set(color='#8ADFD2', linewidth=2)
    ## change the style of fliers and their fill
    for flier in last_eval_boxes['fliers']:
        flier.set(marker='o', color='#2991E7', alpha=0.9)

    # FACE COLORS
    face_color = '#0A3D8C'

    for patch in first_eval_boxes['boxes']:
        patch.set_facecolor(face_color)

    for patch in last_eval_boxes['boxes']:
        patch.set_facecolor(face_color)

    ############## HISTOGRAMS AND DISTRIBUTION ###################

    axs_new[1, 1].xaxis.grid(True, linestyle='dashed', color="#D3D3D3")
    axs_new[1, 0].xaxis.grid(True, linestyle='dashed', color="#D3D3D3")
    axs_new[2, 1].xaxis.grid(True, linestyle='dashed', color="#D3D3D3")
    axs_new[2, 0].xaxis.grid(True, linestyle='dashed', color="#D3D3D3")

    sigma_1 = np.std(general_eval_val_acc[0])
    mu_1 = np.mean(general_eval_val_acc[0])
    n, bins, patches = axs_new[1, 0].hist(np.array(general_eval_val_acc[0]), len(set(general_eval_val_acc[0])) * 2,
                                          density=True, alpha=0.0, edgecolor='black')

    axs_new[2, 0].hist(np.array(general_eval_val_acc[0]),
                                                         len(set(general_eval_val_acc[0])) * 2,
                                                         density=True, alpha=0.7, edgecolor='black')

    bins = (bins[:-1] + bins[1:]) / 2

    pdf_first = 1 / (sigma_1 * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu_1) ** 2 / (2 * sigma_1 ** 2))

    axs_new[1, 0].plot(bins, pdf_first, color='orange', alpha=.6)

    # to ensure pdf and bins line up to use fill_between.

    lower_quartile = np.percentile(np.array(general_eval_val_acc[0]), 25)
    median = np.percentile(np.array(general_eval_val_acc[0]), 50)
    upper_quartile = np.percentile(np.array(general_eval_val_acc[0]), 75)
    IQR = upper_quartile - lower_quartile
    lower_whisker = lower_quartile - 1.5 * IQR
    upper_whisker = upper_quartile + 1.5 * IQR
    upper_whisker = np.compress(general_eval_val_acc[0] <= upper_whisker, general_eval_val_acc[0])
    lower_whisker = np.compress(general_eval_val_acc[0] >= lower_whisker, general_eval_val_acc[0])
    upper_whisker = np.max(upper_whisker)
    lower_whisker = np.min(lower_whisker)
    if upper_whisker < upper_quartile:
        upper_whisker = upper_quartile
    if lower_whisker > lower_quartile:
        lower_whisker = lower_quartile

    bins_1 = np.linspace(lower_whisker, lower_quartile, len(set(general_eval_val_acc[0])) * 2, dtype=float)
    pdf_1 = 1 / (sigma_1 * np.sqrt(2 * np.pi)) * np.exp(-(bins_1 - mu_1) ** 2 / (2 * sigma_1 ** 2))
    bins_2 = np.linspace(upper_whisker, upper_quartile, len(set(general_eval_val_acc[0])) * 2, dtype=float)
    pdf_2 = 1 / (sigma_1 * np.sqrt(2 * np.pi)) * np.exp(-(bins_2 - mu_1) ** 2 / (2 * sigma_1 ** 2))

    # fill from Q1-1.5*IQR to Q1 and Q3 to Q3+1.5*IQR
    axs_new[1, 0].set_ylabel('probability distribution', fontsize=labels_font_size)
    try:
        axs_new[1, 0].fill_between(bins_1, pdf_1, 0, alpha=.6, color='orange')
        axs_new[1, 0].fill_between(bins_2, pdf_2, 0, alpha=.6, color='orange')

        max_bound = max(pdf_first) + max(pdf_first) / 10

        axs_new[1, 0].set_ylim([0, max_bound])

    except ValueError:
        print("Quartile Boundaries are too narrow to be filled")
        # Too few episodes for computing the plot. Axis limits cannot be NaN or Inf
        pass

    axs_new[1, 0].annotate("{:.1f}%".format(100 * norm(mu_1, sigma_1).cdf(lower_quartile)),
                           xy=((lower_whisker + lower_quartile)
                               / 2, 0), ha='center', fontsize=text_font_size)
    axs_new[1, 0].annotate("{:.1f}%".format(100 * (norm(mu_1, sigma_1).cdf(upper_quartile) - norm(mu_1, sigma_1).
                                                   cdf(lower_quartile))), xy=(median, max(pdf_first) / 5), ha='center',
                           fontsize=text_font_size)
    axs_new[1, 0].annotate(
        "{:.1f}%".format(100 * (norm(mu_1, sigma_1).cdf(upper_whisker + upper_quartile) - norm(mu_1, sigma_1).
                                cdf(upper_quartile))), xy=((upper_whisker + upper_quartile) / 2, 0),
        ha='center', fontsize=text_font_size)
    axs_new[1, 0].annotate('q1', xy=(lower_quartile, norm(mu_1, sigma_1).pdf(lower_quartile)), ha='center',
                           fontsize=text_font_size)
    axs_new[1, 0].annotate('q3', xy=(upper_quartile, norm(mu_1, sigma_1).pdf(upper_quartile)), ha='center',
                           fontsize=text_font_size)


    # Other plot

    sigma_2 = np.std(general_eval_val_acc[-1])
    mu_2 = np.mean(general_eval_val_acc[-1])
    n, bins, patches = axs_new[1, 1].hist(np.array(general_eval_val_acc[-1]), len(set(general_eval_val_acc[0])) * 2,
                                          density=True, alpha=0.0, edgecolor='black')

    axs_new[2, 1].hist(np.array(general_eval_val_acc[-1]),
                                                         len(set(general_eval_val_acc[-1])) * 2,
                                                         density=True, alpha=0.7, edgecolor='black')

    bins = (bins[:-1] + bins[1:]) / 2

    pdf_last = 1 / (sigma_2 * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu_2) ** 2 / (2 * sigma_2 ** 2))

    axs_new[1, 1].plot(bins, pdf_last, color='orange', alpha=.6)

    # to ensure pdf and bins line up to use fill_between.

    lower_quartile = np.percentile(np.array(general_eval_val_acc[-1]), 25)
    median = np.percentile(np.array(general_eval_val_acc[-1]), 50)
    upper_quartile = np.percentile(np.array(general_eval_val_acc[-1]), 75)
    IQR = upper_quartile - lower_quartile
    lower_whisker = lower_quartile - 1.5 * IQR
    upper_whisker = upper_quartile + 1.5 * IQR
    upper_whisker = np.compress(general_eval_val_acc[-1] <= upper_whisker, general_eval_val_acc[-1])
    lower_whisker = np.compress(general_eval_val_acc[-1] >= lower_whisker, general_eval_val_acc[-1])
    upper_whisker = np.max(upper_whisker)
    lower_whisker = np.min(lower_whisker)
    if upper_whisker < upper_quartile:
        upper_whisker = upper_quartile
    if lower_whisker > lower_quartile:
        lower_whisker = lower_quartile

    bins_1 = np.linspace(lower_whisker, lower_quartile, len(set(general_eval_val_acc[-1])) * 2, dtype=float)
    pdf_1 = 1 / (sigma_2 * np.sqrt(2 * np.pi)) * np.exp(-(bins_1 - mu_2) ** 2 / (2 * sigma_2 ** 2))
    bins_2 = np.linspace(upper_whisker, upper_quartile, len(set(general_eval_val_acc[-1])) * 2, dtype=float)
    pdf_2 = 1 / (sigma_2 * np.sqrt(2 * np.pi)) * np.exp(-(bins_2 - mu_2) ** 2 / (2 * sigma_2 ** 2))

    axs_new[1, 1].set_ylabel('probability distribution', fontsize=labels_font_size)
    try:
        axs_new[1, 1].fill_between(bins_1, pdf_1, 0, alpha=.6, color='orange')
        axs_new[1, 1].fill_between(bins_2, pdf_2, 0, alpha=.6, color='orange')

        max_bound = max(pdf_last) + max(pdf_last) / 10

        axs_new[1, 1].set_ylim([0, max_bound])
    except ValueError:
        print("Quartile Boundaries are too narrow to be filled")
        # Too few episodes for computing the plot. Axis limits cannot be NaN or Inf
        pass

    axs_new[1, 1].annotate("{:.1f}%".format(100 * norm(mu_2, sigma_2).cdf(lower_quartile)),
                           xy=((lower_whisker + lower_quartile)
                               / 2, 0), ha='center', fontsize=text_font_size)
    axs_new[1, 1].annotate("{:.1f}%".format(100 * (norm(mu_2, sigma_2).cdf(upper_quartile) - norm(mu_2, sigma_2).
                                                   cdf(lower_quartile))), xy=(median, max(pdf_first) / 5), ha='center',
                           fontsize=text_font_size)
    axs_new[1, 1].annotate(
        "{:.1f}%".format(100 * (norm(mu_2, sigma_2).cdf(upper_whisker + upper_quartile) - norm(mu_2, sigma_2).
                                cdf(upper_quartile))), xy=((upper_whisker + upper_quartile) / 2, 0),
        ha='center', fontsize=text_font_size)
    axs_new[1, 1].annotate('q1', xy=(lower_quartile, norm(mu_2, sigma_2).pdf(lower_quartile)), ha='center',
                           fontsize=text_font_size)
    axs_new[1, 1].annotate('q3', xy=(upper_quartile, norm(mu_2, sigma_2).pdf(upper_quartile)), ha='center',
                           fontsize=text_font_size)



    plt.savefig(new_directory + "/" + algorithm_name + str(episodes) +
                "_normal_distribution_first_last_box_plot.pdf", dpi=400)
