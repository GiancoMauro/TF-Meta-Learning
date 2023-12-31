import numpy as np
import scipy as sp
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    """
    Functions that allows the computation of the 0.95 confidence interval on the final tasks
    Args:
        data: list of accuracies values
        confidence: value of confidence that it is wanted to be computed for the simulation

    Returns: mean accuracy and confidence interval

    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def add_noise_images(images):
    """
    function that adds an epsilon of gaussian noise to the images samples
    """
    row, col, ch = images[0].shape
    mean = 0
    var = 0.005
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape((row, col, ch))
    noisy = images + gauss
    return noisy
