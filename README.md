# Federated Learning with CelebA Dataset and MobileNetv2

This repository contains a Python implementation of a federated learning simulation using the CelebA dataset. The focus is on comparing the performance of a MobileNetV2 model trained under both IID (Independent and Identically Distributed) and non-IID (non-Independent and Identically Distributed) data distributions among clients.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Federated Learning Setup](#federated-learning-setup)
- [Performance Evaluation](#performance-evaluation)
- [Usage](#usage)
- [Results](#results)

## Installation

To run this code, you need to install the required dependencies. You can install them using the following commands:

```bash
pip install flwr
pip install -U "flwr[simulation]"
pip install ray
```

Additionally, the code uses PyTorch and torchvision for model training and data handling. Ensure you have these installed as well:

```bash
pip install torch torchvision
```

## Dataset

The CelebA dataset is used for this project. It contains over 200,000 celebrity images with 40 attribute labels. In this implementation, the dataset is split into IID and non-IID subsets among clients to simulate different data distributions.

### Preprocessing

- **Resizing:** Images were resized to 128x128 pixels.
- **Normalization:** Each image was normalized with a mean of `[0.5, 0.5, 0.5]` and a standard deviation of `[0.5, 0.5, 0.5]`.

## Model Architecture

The model used in this project is a modified MobileNetV2:

- **Base Model:** MobileNetV2 with pre-trained weights.
- **Fine-Tuning:** The classifier layer was replaced with a custom fully connected layer with 40 output classes, corresponding to the attributes in the CelebA dataset.
- **Transfer Learning:** The base layers of MobileNetV2 are frozen to leverage pre-trained features, while the classifier is trained on the CelebA attributes.

## Federated Learning Setup

The federated learning framework is implemented using [Flower (flwr)](https://flower.dev/), a popular open-source federated learning library.

### Key Components

- **Clients:** Each client represents a participant in the federated learning process, training the model on their local dataset.
- **Server:** The central server coordinates the training by aggregating model updates from the clients and distributing the global model.

### Data Distribution

- **IID Split:** Data is randomly divided among clients such that each client has a similar distribution of the dataset.
- **Non-IID Split:** Data is divided based on a specific attribute (e.g., "Smiling"), leading to a skewed distribution across clients.

### Training Details

- **Loss Function:** `nn.BCEWithLogitsLoss` is used for training due to the multi-label nature of the CelebA attributes.
- **Optimizer:** Adam optimizer with a learning rate of 0.001.
- **Federated Strategy:** Federated Averaging (FedAvg) is employed to aggregate the model updates from clients.

## Performance Evaluation

The performance of the model is evaluated using classification metrics. The evaluation focuses on comparing the results of models trained with IID and non-IID data distributions.

### Evaluation Metrics

- **Accuracy:** The proportion of correct predictions out of the total predictions.
- **Classification Report:** Provides precision, recall, and F1-score for each attribute.

## Usage

1. Clone the repository.
2. Install the required dependencies.
3. Run the simulation using the provided code.

The simulation can be executed by running the notebook in a Jupyter environment or by converting it to a Python script.

## Results

The performance of the federated learning model is compared between IID and non-IID distributions. The evaluation reports highlight the differences in accuracy and classification metrics under the two data distributions.

### Example Output

```bash
IID Performance Report:
...
Non-IID Performance Report:
...
```


---
