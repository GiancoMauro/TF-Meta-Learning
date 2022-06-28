# TF Meta Learning

A collection of both optimization-based and relation-based meta-learning algorithms developed on TensorFlow.

# Weighting-Injection and Meta-Weighting Nets

ADD INFO PAPER

# Requirements

Python 3.6

Matplotlib 3.3.1,
Numpy 1.19,
Pillow 7.2.0,
Scipy 1.5.4,
Tensorflow 2.4.1

# Data

All experiments are configured for Omniglot with unresized images so 105x105. 
The data can be downloaded from the repository : [Omniglot data](https://github.com/brendenlake/omniglot).

The data in the archive files "images_background.zip" and 
"images_evaluation.zip" should be unzipped and placed directly as sub-folders in ```/data"```.

# Prerequisites
- Run ```pip install -r  requirements.txt ```
- Download and store Omniglot data in the ```/data``` folder
- Create a folder ```/results``` to store experiment results (both plots and saved models). 
Results is also automatically created after first experiment completion.

# Code execution

From terminal (Once defined the configurations): ```cd ./algorithms; python "ALGORITHM.py"``` where ```"```

# Reference algorithms
Reptile:
Nichol, Alex, Joshua Achiam, and John Schulman. ["On first-order meta-learning algorithms."](https://arxiv.org/abs/1803.02999), arXiv preprint arXiv:1803.02999). Partial re implementation of: ADMoreau. ["Few-Shot learning with Reptile."](https://keras.io/examples/vision/reptile/).

MAML (1st and 2nd order): 
Finn, Chelsea, Pieter Abbeel, and Sergey Levine. ["Model-agnostic meta-learning for fast adaptation of deep networks."](https://arxiv.org/abs/1703.03400), International conference on machine learning. PMLR, 2017.

MAML+MSL+CA+DA: 
Antoniou, Antreas, Harri Edwards, and Amos Storkey. ["How to train your MAML."](https://arxiv.org/abs/1810.09502), Seventh International Conference on Learning Representations. 2019.
The following contributes from the paper have been implemented in this tensorflow version of MAML:
1.  Multi-Step Loss Optimization (MSL)
2.  Cosine Annealing of Meta-Optimizer Learning Rate (CA)
3.  Derivative-Order Annealing (DA)