import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FCN(nn.Module):
    """
    Fully connected neural networks.

    """
    def __init__(self, input_size, output_size):
        super(FCN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

X = np.load('toy_X.npy')
y = np.load('toy_y.npy').reshape(-1, 1)

X = torch.from_numpy(X).float().to(device)
y = torch.from_numpy(y).float().to(device)

model = FCN(2, 1)
#K = torch.Tensor([[0, 0]])
#model.fc[0].weight = torch.nn.Parameter(K)
#K = torch.Tensor([[0]])
#model.fc[0].bias = torch.nn.Parameter(K)
print(model.state_dict())

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.05)
criterion = nn.BCELoss()

for e in range(100):
    model.train()
    optimizer.zero_grad()

    output = model(X)

    loss = criterion(output, y)
    if e % 10 == 0:
        print("epoch={}, loss={}".format(e, loss.item()))
    loss.backward()
    optimizer.step()

    #print(output.cpu().detach().numpy())

    threshold = torch.tensor([0.5]).to(device)
    results = (output>threshold).float()*1
    #print(output.cpu().detach().numpy().reshape(-1, 1), results.cpu().detach().numpy().reshape(-1, 1))

model.eval()
output = model(X)
threshold = torch.tensor([0.5]).to(device)
results = (output>threshold).float()*1
print(output)
print(model.state_dict())