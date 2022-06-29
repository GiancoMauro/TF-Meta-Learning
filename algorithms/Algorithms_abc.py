from abc import ABC


class AlgorithmsABC(ABC):
    """
        Core Implementation of the Algorithms
        """

    def __init__(self, n_shots, n_ways, n_episodes, n_query, n_tests, train_dataset, test_dataset,
                 n_repeat, n_box_plots, eval_inter, beta_1, beta_2, xbox_multiples):
        self.beta1 = beta_1
        self.beta2 = beta_2
        self.episodes = n_episodes
        self.eval_interval = eval_inter
        self.experiments_num = n_repeat
        self.num_box_plots = n_box_plots
        self.xbox_multiples = xbox_multiples
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.n_ways = n_ways
        self.support_train_shots = n_shots
        self.query_shots = n_query

        self.test_shots = n_tests
