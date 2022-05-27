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

csv = "Epoch,Accuracy (Training Set),Accuracy (Validation Set)\n"
for epoch in range(1000):
    train(net, train_set_w, criterion, optimizer)
    accuracy_t = validate(net, train_set_w)
    accuracy_v = validate(net, val_set_w)

    if (epoch + 1) % 100 == 0 or epoch == 0:
        print("Finished epoch {}\nAccuracy training set: {}%\nAccuracy validation set: {}%\n---------------------------"
              .format(epoch + 1, int(accuracy_t), int(accuracy_v)))
    csv += "{},{},{}\n".format(epoch + 1, accuracy_t, accuracy_v)

f = open("results.csv", "w+")
f.write(csv)

print("Training finished!")
