!pip install flwr
!pip install -U "flwr[simulation]"
!pip install ray

import ray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, random_split
import flwr as fl
from sklearn.metrics import classification_report
from torch.utils.data import Subset
import time
import multiprocessing
#multiprocessing.set_start_method("spawn")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load CelebA dataset
celeba_dataset = CelebA(root="./data", split="all", download=True, transform=transform)

NUM_CLIENTS = 50

def iid_split(dataset, num_clients):
    """Splits the dataset into IID subsets for each client."""
    data_len = len(dataset)
    indices = np.random.permutation(data_len)
    split_indices = np.array_split(indices, num_clients)
    return [torch.utils.data.Subset(dataset, indices) for indices in split_indices]

def non_iid_split(dataset, num_clients):
    """Splits the dataset into non-IID subsets for each client."""
    # Assuming the label we use is binary (like gender, smiling, etc.)
    targets = np.array(dataset.attr[:, 20])  # Attribute index 20 for "Smiling"
    indices = np.argsort(targets)
    split_indices = np.array_split(indices, num_clients)
    return [torch.utils.data.Subset(dataset, indices) for indices in split_indices]


iid_datasets = iid_split(celeba_dataset, NUM_CLIENTS)
non_iid_datasets = non_iid_split(celeba_dataset, NUM_CLIENTS)

class CustomMobileNetV2(nn.Module):
    def __init__(self, num_classes=40):
        super(CustomMobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = CustomMobileNetV2(num_classes=40)

class CelebAClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config={}):
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.dtype).to(param.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for images, labels in self.train_loader:
            #print("Shape of labels:", labels.shape)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels.float())
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                predicted = (outputs > 0.5).float()  # Threshold predictions at 0.5
                total += labels.size(0) * labels.size(1)  # Count total predictions
                correct += (predicted == labels).sum().item()
        return float(correct) / total, len(self.test_loader.dataset), {}

# IID and non-IID clients
clients_iid = [CelebAClient(model, iid_dataset, iid_dataset) for iid_dataset in iid_datasets]
clients_non_iid = [CelebAClient(model, non_iid_dataset, non_iid_dataset) for non_iid_dataset in non_iid_datasets]

def client_fn(cid: str) -> fl.client.Client:
    return clients_iid[int(cid)].to_client()

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=10,
    min_evaluate_clients=10,
    min_available_clients=NUM_CLIENTS,
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy,
)

from sklearn.metrics import classification_report

def evaluate_performance(client, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = client.model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return classification_report(all_labels, all_preds)

# Compare across demographic groups
iid_report = evaluate_performance(clients_iid[0], celeba_dataset)
non_iid_report = evaluate_performance(clients_non_iid[0], celeba_dataset)

print("IID Performance Report:\n", iid_report)
print("Non-IID Performance Report:\n", non_iid_report)

def evaluate_performance(client, dataset, num_samples=100):
    # Create a subset of the dataset containing only the first `num_samples`
    subset = Subset(dataset, range(num_samples))
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = client.model(images)
            #predicted = torch.argmax(outputs, dim=1)  # Multiclass, select index with highest probability
            predicted = torch.where(outputs > 0.5, torch.tensor(1), torch.tensor(0))
            #print(outputs.size(), '\n', predicted, '\n', labels)
            #time.sleep(5)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #all_preds_multilabel = np.zeros((len(all_preds), outputs.shape[1]))
    #all_preds_multilabel[np.arange(len(all_preds)), all_preds] = 1
    #return classification_report(all_labels, all_preds_multilabel)  # Use multilabel-indicator predictions
    return classification_report(all_labels, all_preds)  # Use multilabel-indicator predictions

iid_report = evaluate_performance(clients_iid[0], celeba_dataset, num_samples=100)
non_iid_report = evaluate_performance(clients_non_iid[0], celeba_dataset, num_samples=100)

print("IID Performance Report:\n", iid_report)
print("Non-IID Performance Report:\n", non_iid_report)
