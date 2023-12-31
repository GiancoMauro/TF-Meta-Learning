from abc import ABC

class AlgorithmsABC(ABC):
    """
        Core Implementation of the Algorithms
        """

    def __init__(self, alg, n_shots, n_ways, n_episodes, n_query, n_tests, train_dataset, test_dataset,
                 n_repeats, n_box_plots, eval_step, beta1, beta2, xbox_multiples, n_fin_episodes, results_dir):
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
        self.results_dir = results_dir

        self.n_ways = n_ways
        self.support_train_shots = n_shots
        self.query_shots = n_query

        self.test_shots = n_tests