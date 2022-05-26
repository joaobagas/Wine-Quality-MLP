from torch import nn, optim

from model import Net
from torch.utils.data import random_split
from functions import *

input_path_w = "dataset/winequality-white.csv"
dataset_w = load_data(input_path_w)
train_set_w, val_set_w = random_split(dataset_w, [3918, 980])  # 4898 = 3918 + 980 (80:20)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-4)

print("Training...")

csv = "Epoch,Loss (Training Set),Loss (Validation Set)\n"
for epoch in range(1000):
    train(net, train_set_w, criterion, optimizer)
    loss_t = validate(net, train_set_w, criterion)
    loss_v = validate(net, val_set_w, criterion)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print("{}%".format((epoch+1)/10))
        print("Finished epoch {}! Loss training set:{}! Loss validation set:{}!".format(epoch + 1, loss_t, loss_v))
    csv += "{},{},{}\n".format(epoch + 1, loss_t, loss_v)

f = open("results.csv", "w+")
f.write(csv)

print("Training finished!")
