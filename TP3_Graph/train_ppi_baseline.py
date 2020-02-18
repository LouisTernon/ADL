import argparse
from os import path

import numpy as np
import torch
import torch.nn.functional as F
from dgl import batch
from dgl.data.ppi import LegacyPPIDataset
from dgl.nn.pytorch import GraphConv,  GATConv
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader

# Fix seed
np.random.seed(0)
torch.manual_seed(0)

MODEL_STATE_FILE = path.join(path.dirname(path.abspath(__file__)), "model_state.pth")


class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, activation=None, alpha=.2, concat_heads=True):
        super(GATLayer, self).__init__()
        self.linear_func = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attention_func = nn.Linear(2 * out_feats, num_heads, bias=False)
        self.alpha = alpha
        self.num_heads = num_heads
        self.activation = activation
        self.out_features = out_feats
        self.concat_heads = concat_heads

    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src['z'], edges.dst['z']], dim=2)
        src_e = torch.stack([concat_z[:, i].matmul(self.attention_func.weight[i, :].t()) for i in range(self.num_heads)]).t()
        src_e = F.leaky_relu(src_e, negative_slope=self.alpha)
        return {'e': src_e}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        a = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(a.unsqueeze(-1) * nodes.mailbox['z'], dim=1)
        if self.concat_heads:
            h = h.view(-1, self.num_heads * self.out_features)
        else:
            h = torch.mean(h, dim=1)
        return {'h': h}

    def forward(self, graph, h):
        # graph = graph.local_var()
        z = self.linear_func(h)
        graph.ndata['z'] = z.view((-1, self.num_heads, self.out_features))
        graph.apply_edges(self.edge_attention)
        graph.update_all(self.message_func, self.reduce_func)
        h = graph.ndata.pop('h')
        if self.activation is not None:
            h = self.activation(h)
        return h

class AttentionModel(nn.Module):

    def __init__(self, g, n_layers, input_size, hidden_size, output_size, activation, **kwargs):
        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(input_size, hidden_size, activation=activation, num_heads=kwargs["num_heads"]))
        for i in range(n_layers - 1):
            self.layers.append(GATLayer(hidden_size, hidden_size, activation=activation, num_heads=kwargs["num_heads"]))
        self.layers.append(GATLayer(hidden_size*kwargs["num_heads"], output_size, num_heads=1))

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)
            outputs.squeeze_()
        return outputs


class SparseGATConv(nn.Module):

    def __init__(self, in_features, out_features, num_heads=1, activation= None, average_head=False, bias=False, alpha=.2):
        super(SparseGATConv, self).__init__()
        self._in_f = in_features
        self._out_f = out_features
        self._activation = activation
        self._num_heads = num_heads
        self._average_head = average_head
        self._bias = bias
        self.lrelu = nn.LeakyReLU(alpha)

class OwnGATConv(nn.Module):

    def __init__(self, in_features, out_features, num_heads=1, activation= None, average_head=False, bias=False, alpha=.2):
        super(OwnGATConv, self).__init__()
        self._in_f = in_features
        self._out_f = out_features
        self._activation = activation
        self._num_heads = num_heads
        self._average_head = average_head
        self._bias = bias
        self.lrelu = nn.LeakyReLU(alpha)


        self.w = nn.Parameter(torch.Tensor(self._out_f*self._num_heads, self._in_f))
        if self._bias: self.bias = nn.Parameter(torch.Tensor(self._out_f*self._num_heads))
        self.a = nn.Parameter(torch.Tensor(2*self._out_f*self._num_heads, 1))

        # Initialization
        self.reset_parameters()

    def reset_parameters(self):
            nn.init.xavier_uniform_(self.w, gain=nn.init.calculate_gain('relu'))
            if self._bias: nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.a, gain=nn.init.calculate_gain('relu'))

    def forward(self, graph, inputs):
        if self._bias: _h = torch.addmm(self.bias, inputs, self.w.t())
        else:  _h = torch.mm(inputs, self.w.t())

        nodes = _h.size()[0]
        # compute a
        _concat_f = torch.cat([_h.repeat(1, nodes).view(nodes * nodes, -1), _h.repeat(nodes, 1)], dim=1).view(nodes, -1, 2 * self._out_f)
        _key = self.lrelu(torch.matmul(_concat_f, self.a).squeeze(2))

        _masked_attention = torch.where(graph.adjacency_matrix().to_dense() > 0, _key, torch.zeros_like(_key))
        _masked_attention = F.softmax(_masked_attention, dim=1)
        _h_prime = torch.matmul(_masked_attention, _h)

        if not self._average_head:
            return self._activation(_h_prime)
        else:
            return _h_prime.view((-1, nodes, self._out_f)).mean(axis=0)

class CustomGraphConv(nn.Module):

    def __init__(self, in_features, out_features, activation=None):
        super(CustomGraphConv, self).__init__()
        self._in_f = in_features
        self._out_f = out_features
        self._activation = activation

        self.weight = nn.Parameter(torch.rand((self._in_f, self._out_f))/100-.005)
        self.bias = nn.Parameter(torch.rand((self._out_f))/100-.005)

    def forward(self, graph, inputs):
        A = graph.adjacency_matrix()
        A_ = torch.eye(A.size()[0]) + A
        D = torch.diag(torch.sum(A_, axis=1)**(-1/2))
        output = torch.mm(torch.mm(torch.mm(torch.mm(D, A_), D), inputs), self.weight) + self.bias
        if self._activation is not None: output = self._activation(output)

        return output

class CustomGraphModel(nn.Module):

    def __init__(self, g, n_layers, input_size, hidden_size, output_size, nonlinearity, **kwargs):
        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(CustomGraphConv(input_size, hidden_size, activation=nonlinearity))
        for i in range(n_layers - 1):
            self.layers.append(CustomGraphConv(hidden_size, hidden_size, activation=nonlinearity))
        self.layers.append(CustomGraphConv(hidden_size, output_size))

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)
        return outputs

class BasicGraphModel(nn.Module):

    def __init__(self, g, n_layers, input_size, hidden_size, output_size, nonlinearity, **kwargs):
        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_size, hidden_size, activation=nonlinearity))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_size, hidden_size, activation=nonlinearity))
        self.layers.append(GraphConv(hidden_size, output_size))

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)
        return outputs

class BasicGATModel(nn.Module):

    def __init__(self, g, n_layers, input_size, hidden_size, output_size, nonlinearity, **kwargs):
        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(input_size, hidden_size, activation=None, num_heads=kwargs["num_heads"]))
        for i in range(n_layers - 1):
            self.layers.append(GATConv(hidden_size, hidden_size, activation=None, num_heads=kwargs["num_heads"]))
        self.layers.append(GATConv(hidden_size*kwargs["num_heads"], output_size, num_heads=1))

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)
            outputs.squeeze_()
        return outputs

class OwnGATModel(nn.Module):

    def __init__(self, g, n_layers, input_size, hidden_size, output_size, num_heads):
        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(OwnGATConv(input_size, hidden_size, num_heads))
        for i in range(n_layers - 1):
            self.layers.append(OwnGATConv(input_size, hidden_size, num_heads))
        self.layers.append(OwnGATConv(input_size, hidden_size, num_heads, average_head=True))

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)
            outputs.squeeze_()
        return outputs


MODELS = {"baseline": BasicGraphModel, "attention": BasicGATModel, "custom": CustomGraphModel, "own_attention":OwnGATModel,
          "message_passing_attention":AttentionModel}


def main(args):
    # create the dataset
    train_dataset, test_dataset = LegacyPPIDataset(mode="train"), LegacyPPIDataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    n_features, n_classes = train_dataset.features.shape[1], train_dataset.labels.shape[1]

    # create the model, loss function and optimizer
    device = torch.device("cpu" if args.gpu < 0 else "cuda:" + str(args.gpu))
    model_kwargs = {"num_heads":5}
    model = MODELS[args.model](g=train_dataset.graph, n_layers=2, input_size=n_features,
                               hidden_size=256, output_size=n_classes, activation=None, **model_kwargs).to(device)
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train and test
    if args.mode == "train":
        train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset)
        torch.save(model.state_dict(), MODEL_STATE_FILE)
    model.load_state_dict(torch.load(MODEL_STATE_FILE))
    return test(model, loss_fcn, device, test_dataloader)


def train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=10, threshold=1e-3)
    loss_data=np.inf
    for epoch in range(args.epochs):
        scheduler.step(loss_data)
        model.train()
        losses = []
        for batch, data in enumerate(train_dataloader):
            subgraph, features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            model.g = subgraph
            for layer in model.layers:
                layer.g = subgraph
            logits = model(features.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        print("Epoch {:05d} | Loss: {:.4f} | Lr : {:.4e}".format(epoch + 1, loss_data, optimizer.param_groups[0]['lr']))

        if epoch % 5 == 0:
            scores = []
            for batch, test_data in enumerate(test_dataset):
                subgraph, features, labels = test_data
                features = torch.tensor(features).to(device)
                labels = torch.tensor(labels).to(device)
                score, _ = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn)
                scores.append(score)
            print("F1-Score: {:.4f} ".format(np.array(scores).mean()))


def test(model, loss_fcn, device, test_dataloader):
    test_scores = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, features, labels = test_data
        features = features.to(device)
        labels = labels.to(device)
        test_scores.append(evaluate(features, model, subgraph, labels.float(), loss_fcn)[0])
    mean_scores = np.array(test_scores).mean()
    print("F1-Score: {:.4f}".format(np.array(test_scores).mean()))
    return mean_scores


def evaluate(features, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph
        output = model(features.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average="micro")
        return score, loss_data.item()


def collate_fn(sample):
    graphs, features, labels = map(list, zip(*sample))
    graph = batch(graphs)
    features = torch.from_numpy(np.concatenate(features))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, features, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model", type=str, default="baseline")
    args = parser.parse_args()
    main(args)
