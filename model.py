import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(11, 22),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(22, 22),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(22, 11),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def save(net, optimizer, epoch, loss):
    path = "checkpoint/model{}.pt".format(epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load():
    model = Net()
    path = "checkpoint/model.pt"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, epoch, loss
