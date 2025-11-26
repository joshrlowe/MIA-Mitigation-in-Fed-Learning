# Evaluating Mitigation Strategies for Membership Inference Attacks in Federated Learning

Josh Lowe – Master of Science in Computer Science Student

University of Central Florida – Orlando, Florida, USA

[jo201760@ucf.edu](mailto:jo201760@ucf.edu)

This repository implements a federated learning machine learning environment to train Wide ResNet 28-4 on CIFAR-100, and evaluates differential privacy as a defense against membership inference attacks.

This project is built using PyTorch, [Flower](https://flower.ai/), which simulates federated learning with multiple clients, [Opacus](https://opacus.ai/) to toggle differential privacy, [Adversarial Robustnes Toolbox (ART)](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/) to conduct membership inference attacks on the federated learning models for academic purposes.

To implement Wide ResNet 28-4, I referred to the PyTorch implementation in the [WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch/tree/master) for guidance, which implements the paper "Wide Residual Networks" by Sergey Zagoruyko and Nikos Komodakis.

## Functions of This Repository by File

`server_app.py`: Initializes the global Wide ResNet model and orchestrates a federated learning workflow.

`client_app.py`: Trains the Wide ResNet model locally on part of the CIFAR-100 dataset, and can apply differential privacy if differential privacy is toggled, then returns updated model weights to `server_app.py`.

`task.py`: Defines Wide ResNet and its data controls, such as loading data, training a model, and splitting data equally among classes.

`mia.py`: Runs a membership inference attack against a specified model and returns classification metrics to the user to see how well their model withstands membership inference attacks.

This project is designed to run in a Google Colab environment to leverage GPUs during training. To view this project, visit the associated [Google Colab Notebook](https://colab.research.google.com/drive/19Kc1mIwP4wg_A8Z68-to10sPrkfF7OL_?usp=sharing).

The above Colab notebook is view only. If you want to run your own copy of this project, complete the following steps:

1. With the above Colab notebook open, go to File -> Save a copy in Drive.
2. Open your Google Drive, navigate to the saved copy, open it.
3. If subscribed to Colab Pro, go to Runtime -> Change Runtime Type -> Select "A100 GPU", Toggle High-RAM "On". This speeds up training by using a strong GPU in the background.
4. Click "Run All" to run all of the cells.
