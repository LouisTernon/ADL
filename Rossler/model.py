from rossler_map import RosslerMap
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Module
import torch.nn.functional as F
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


a=0.2
b=0.2
c=5.7


def train(num_epochs, batch_size, criterion, optimizer, model, dataset):
    train_error = []
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        epoch_average_loss = 0.0
        for (X_batch, y_real) in train_loader:
            y_pre = model(X_batch.float())
            loss = criterion(y_pre, y_real.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_average_loss += loss.item() * batch_size / len(dataset)
        train_error.append(epoch_average_loss)
        print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, epoch_average_loss))
    return train_error


def simulate_model(model, init_point, epochs):
    model.eval()
    with torch.no_grad():
        y_pred = [model(torch.tensor(init_point).float()).numpy()]
        for k in range(epochs):
            y_pred.append(model(torch.tensor(y_pred[-1]).float()).numpy())
    return np.array(y_pred)


class l1_penalised(Module):
    def __init__(self, lambd, reduction = 'mean'):
        super(l1_penalised, self).__init__()
        self.lambd = lambd
        self.reduction = reduction

    def forward(self, input, target):
        dw_dt_hat = torch.tensor([1,0,0]) * (- input[1] - input[2])
        dw_dt_hat += torch.tensor([0,1,0]) * (input[0] + a * input[1])
        dw_dt_hat += torch.tensor([0,0,1]) * (b + input[2] * (input[0] - c))

        dw_dt = torch.tensor([1,0,0]) * (- target[1] - target[2])
        dw_dt += torch.tensor([0,1,0]) * (target[0] + a * target[1])
        dw_dt += torch.tensor([0,0,1]) * (b + target[2] * (target[0] - c))

        return F.l1_loss(input, target, reduction=self.reduction) + self.lambd * F.l1_loss(dw_dt_hat, dw_dt, reduction=self.reduction)

class l2_penalised(Module):
    def __init__(self, lambd, reduction = 'mean'):
        super(l2_penalised, self).__init__()
        self.lambd = lambd
        self.reduction = reduction


    def forward(self, input, target):
        dw_dt_hat = torch.tensor([1,0,0]) * (- input[1] - input[2])
        dw_dt_hat += torch.tensor([0,1,0]) * (input[0] + a * input[1])
        dw_dt_hat += torch.tensor([0,0,1]) * (b + input[2] * (input[0] - c))

        dw_dt = torch.tensor([1,0,0]) * (- target[1] - target[2])
        dw_dt += torch.tensor([0,1,0]) * (target[0] + a * target[1])
        dw_dt += torch.tensor([0,0,1]) * (b + target[2] * (target[0] - c))

        return F.mse_loss(input, target, reduction=self.reduction) + self.lambd * F.mse_loss(dw_dt_hat, dw_dt, reduction=self.reduction)


class l1_penalised_jacobian(Module):
    def __init__(self, lambd, reduction = 'mean'):
        super(l1_penalised_jacobian, self).__init__()
        self.lambd = lambd
        self.reduction = reduction

    def forward(self, input, target):
        A_hat = torch.tensor([[0,-1,-1], [1, a, 0], [0, 0, 0]])
        A_hat += torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]]) * input[2]
        A_hat += torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]]) * (input[0] - c)
        J_hat = (A_hat*delta_t).exp()


        A = torch.tensor([[0,-1,-1], [1, a, 0], [0, 0, 0]])
        A += torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]]) * target[2]
        A += torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]]) * (target[0] - c)
        J = (A*delta_t).exp()

        return F.l1_loss(input, target, reduction=self.reduction) + self.lambd * F.l1_loss(J_hat, J, reduction=self.reduction)

class l2_penalised_jacobian(Module):
    def __init__(self, lambd, reduction = 'mean'):
        super(l2_penalised_jacobian, self).__init__()
        self.lambd = lambd
        self.reduction = reduction


    def forward(self, input, target):
        A_hat = torch.tensor([[0,-1,-1], [1, a, 0], [0, 0, 0]])
        A_hat += torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]]) * input[2]
        A_hat += torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]]) * (input[0] - c)
        J_hat = (A_hat*delta_t).exp()


        A = torch.tensor([[0,-1,-1], [1, a, 0], [0, 0, 0]])
        A += torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]]) * target[2]
        A += torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]]) * (target[0] - c)
        J = (A*delta_t).exp()


        return F.mse_loss(input, target, reduction=self.reduction) + self.lambd * F.mse_loss(J_hat, J, reduction=self.reduction)







if __name__ == '__main__':

    model = nn.Sequential(
        torch.nn.Linear(3, 100),
        torch.nn.Linear(100,100),
        torch.nn.Tanh(),
        torch.nn.Linear(100,100),
        torch.nn.Tanh(),
        torch.nn.Linear(100,3))


    delta_t = 1e-2
    ROSSLER_MAP = RosslerMap(delta_t=delta_t)
    INIT = np.array([-5.75, -1.6, 0.02])

    trajs, t = ROSSLER_MAP.full_traj(100000, INIT)



    optimizer = torch.optim.Adam(model.parameters())


    #criterion = torch.nn.MSELoss()
    #criterion = l1_penalised(RosslerMap.a, RosslerMap.b, RosslerMap.c, lambd=1)


    x = torch.tensor(trajs[:-1])
    y = torch.tensor(trajs[1:])

    training_set = TensorDataset(x[:8000], y[:8000])


    #train(5, 64, criterion, optimizer, model, training_set)

    y_true = trajs[8001:]
    x_init = trajs[8000]
    y_pred = simulate_model(model, INIT, len(y_true))


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.plot(y_pred[:,0], y_pred[:,1], y_pred[:,2], lw=.1)
    ax.plot(trajs[:,0], trajs[:,1], trajs[:,2], lw=.1)

    plt.show()
