from torch import nn, optim

from model import Net
from torch.utils.data import random_split
from functions import *

input_path_w = "dataset/winequality-white.csv"
dataset_w = load_data(input_path_w)
train_set_w, val_set_w = random_split(dataset_w, [3918, 980])  # 4898 = 3918 + 980 (80:20)

net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=3e-3)

print("Training...")

csv = ""
for epoch in range(1000):
    train(net, train_set_w, criterion, optimizer)
    loss = validate(net, val_set_w, criterion)
    print("Finished epoch {}! Loss:{}!".format(epoch + 1, loss))
    csv += "{},{}\n".format(epoch + 1, loss)

f = open("results.csv", "w+")
f.write(csv)

print("Training finished!")
