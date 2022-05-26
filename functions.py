import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def load_data(input_path):
    data = pd.read_csv(input_path, sep=";")

    input_cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                  "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
    output_cols = ["quality"]

    inputs_array = data[input_cols].to_numpy()
    targets_array = data[output_cols].to_numpy()

    inputs = torch.from_numpy(inputs_array).type(torch.float)
    targets = torch.from_numpy(targets_array).type(torch.float)

    one_h_targets = F.one_hot(targets.long(), num_classes=11)

    return TensorDataset(inputs, one_h_targets)


def train(net, train_set, criterion, optimizer):
    train_loader = DataLoader(train_set, 20, shuffle=True)
    net.train()

    for batch in train_loader:
        inputs, results = batch
        results = results.to(torch.float32)
        size = results.size()[0]
        outputs = net.forward(inputs)
        loss = criterion(outputs.reshape(size, 11), results.reshape(size, 11))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def validate(net, val_set, criterion):
    val_loader = DataLoader(val_set, 20)
    net.eval()
    losses, times = 0, 0

    for batch in val_loader:
        inputs, results = batch
        outputs = net.forward(inputs)
        size = results.size()[0]
        losses += criterion(outputs.reshape(size, 11).float(), results.reshape(size, 11).float())
        times += 1

    return losses/times

def calculate_accuracy(inputs, outputs):
    print("")