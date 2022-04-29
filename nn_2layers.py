import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FCN(nn.Module):
    """
    Fully connected neural networks.

    """
    def __init__(self, input_size, hidden_dim, output_size):
        super(FCN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

class NeuralNet():
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.05, X_test=None, y_test=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = FCN(input_dim, hidden_dim, output_dim)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

        self.X_test = X_test
        self.y_test = y_test

        self.model = self.model.to(device)

    def fit(self, X, y, num_iter=100):

        X = torch.from_numpy(X).float().to(device)
        y = torch.from_numpy(y).float().to(device)

        for e in range(num_iter):
            self.model.train()
            self.optimizer.zero_grad()

            self.model = self.model.to(device)
            output = self.model(X)

            loss = self.criterion(output, y)
            threshold = torch.tensor([0.5]).to(device)
            results = (output>threshold).float()*1
            correct_num = (results == y).float().sum()
            train_acc = torch.round(correct_num / len(y) * 100)
            if e % 10 == 0:
                print("Train: epoch: {}, loss: {:.3f}, acc: {:.3f}".format(e, loss.item(), train_acc))

                if self.X_test is not None and self.y_test is not None:
                    test_loss, test_acc = self.score(self.X_test, self.y_test)
                    print("Test: loss: {:.3f}, acc: {:.3f}".format(test_loss, test_acc))

            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        self.model = self.model.to(device)
        self.model.eval()

        X = torch.from_numpy(X).float().to(device)

        self.optimizer.zero_grad()
        output = self.model(X)

        return output

    def score(self, X, y):
        output = self.predict(X)

        y = torch.from_numpy(y).float().to(device)
        loss = self.criterion(output, y)

        threshold = torch.tensor([0.5]).to(device)
        results = (output>threshold).float()*1

        correct_num = (results == y).float().sum()
        test_acc = torch.round(correct_num / len(y) * 100)

        return loss, test_acc

if __name__ == "__main__":
    """
    X = np.load('toy_X.npy')
    y = np.load('toy_y.npy').reshape(-1, 1)

    neuralnet = NeuralNet(2, 2, 1, lr=0.05, X_test=None, y_test=None)
    K = torch.Tensor([[0.6062185883274536, -0.02400056629504066], [0.9083340590954418, 1.0611354924277814]])
    neuralnet.model.fc[0].weight = torch.nn.Parameter(K)
    K = torch.Tensor([[0.0, 0.0]])
    neuralnet.model.fc[0].bias = torch.nn.Parameter(K)

    K = torch.Tensor([[5.011998701374626, -1.1431551718763109]])
    neuralnet.model.fc[2].weight = torch.nn.Parameter(K)
    K = torch.Tensor([[-1.6666666666666643]])
    neuralnet.model.fc[2].bias = torch.nn.Parameter(K)
    """

    X = np.load('titanic_X_train.npy')
    y = np.load('titanic_y_train.npy').reshape(-1, 1)

    neuralnet = NeuralNet(5, 5, 1, lr=0.05, X_test=None, y_test=None)
    K = torch.Tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [5.853552954208808e-10, -1.2354092344902669e-09, -3.017959160808027e-10, -1.5164403066592058e-10, -1.7486119905001385e-12]])
    neuralnet.model.fc[0].weight = torch.nn.Parameter(K)
    K = torch.Tensor([[0.0, -99.99999999768139]])
    neuralnet.model.fc[0].bias = torch.nn.Parameter(K)

    K = torch.Tensor([[1.855836391486696, -399709892149.4715]])
    neuralnet.model.fc[2].weight = torch.nn.Parameter(K)
    K = torch.Tensor([[-20.695247345136565]])
    neuralnet.model.fc[2].bias = torch.nn.Parameter(K)


    neuralnet.fit(X, y)

    print(neuralnet.score(X, y))

    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    neuralnet.model.eval()
    neuralnet.model = neuralnet.model.to(device)
    output = neuralnet.model(X)
    threshold = torch.tensor([0.5]).to(device)
    results = (output>threshold).float()*1

    correct_num = (results == y).float().sum()
    train_acc = correct_num / len(y) * 100
    #print(results)
    print(train_acc)

    print(neuralnet.model.state_dict())

# Check MP
"""
K = torch.Tensor([[0.0, 0.0], [1.1314813736338398, -1.13148137363384]])
model.fc[0].weight = torch.nn.Parameter(K)
K = torch.Tensor([[0.0, 2.26296274726768]])
model.fc[0].bias = torch.nn.Parameter(K)

K = torch.Tensor([[0.2385, 1.7171]])
model.fc[1].weight = torch.nn.Parameter(K)
K = torch.Tensor([[-2.8286]])
model.fc[1].bias = torch.nn.Parameter(K)
"""

# Check SP
"""
K = torch.Tensor([[-0.1458,  0.0463], [1.8124, -1.0703]])
model.fc[0].weight = torch.nn.Parameter(K)
K = torch.Tensor([[-0.4035,  0.6006]])
model.fc[0].bias = torch.nn.Parameter(K)

K = torch.Tensor([[0.0, 1.20785103170613]])
model.fc[1].weight = torch.nn.Parameter(K)
K = torch.Tensor([[-3.0000000000002]])
model.fc[1].bias = torch.nn.Parameter(K)
"""

#print(model.state_dict())

#model = model.to(device)

#optimizer = optim.Adam(model.parameters(), lr=0.05)
#criterion = nn.BCELoss()

"""
for e in range(100):
    model.train()
    optimizer.zero_grad()

    output = model(X)

    loss = criterion(output, y)
    if e % 1 == 0:
        print("epoch={}, loss={}".format(e, loss.item()))
    loss.backward()
    optimizer.step()

    #print(output.cpu().detach().numpy())

    threshold = torch.tensor([0.5]).to(device)
    results = (output>threshold).float()*1
    #print(output.cpu().detach().numpy().reshape(-1, 1), results.cpu().detach().numpy().reshape(-1, 1))
"""

#model.eval()
#output = model(X)
#threshold = torch.tensor([0.5]).to(device)
#results = (output>threshold).float()*1
#print(output)
#print(model.state_dict())