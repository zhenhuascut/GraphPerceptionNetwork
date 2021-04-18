import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_geometric.nn import GINConv
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_add


class GPNConv(torch.nn.Module):
    def __init__(self, nn, input_size=None, hidden_size=None, gates_num=4):
        super(GPNConv, self).__init__()
        # self.nn = NNLinear(input_size, hidden_size, gates_num)
        if nn is not None:
            self.nn = nn
        else:
            self.nn = torch.nn.Linear(input_size, hidden_size*gates_num, bias=True)

    def forward(self, x, edge_index):
        # x = x.unsqueeze(-1) if x.dim()==1 else x
        row, col = edge_index
        out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        out = x + out
        out = self.nn(out)
        return out


class GINConv(torch.nn.Module):
    def __init__(self, nn_module, input_size, hidden_size, gates_num=4):
        super(GINConv, self).__init__()
        # self.nn = NNLinear(input_size, hidden_size, gates_num)
        if nn_module is None:
            self.nn = torch.nn.Linear(input_size, hidden_size * gates_num, bias=True)

    def forward(self, x, edge_index):
        """"""
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        row, col = edge_index
        out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        out = x + out
        out = self.nn(out)
        return F.relu(out)

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class GPN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, return_middle_feature=False):
        super(GPN, self).__init__()
        self.gpn_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        for layer in range(num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            gpn_conv = GPNConv(mlp, 0)
            self.gpn_layers.append(gpn_conv)

            batch_norm = nn.BatchNorm1d(hidden_dim)
            self.batch_norms.append(batch_norm)

        self.return_middle_feature = return_middle_feature

        self.linears_prediction = torch.nn.ModuleList()
        middle_dim = 32
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, middle_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, middle_dim))

        self.output_layer = nn.Linear(middle_dim, output_dim)

    def forward(self, x, A, batch_index):
        h = x
        hidden_rep = [h]
        for layer in range(self.num_layers - 1):
            h = self.gpn_layers[layer](h, A)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            hidden_rep.append(h)

        output_h = 0
        for layer, h in enumerate(hidden_rep):
            if not(layer==0 or layer == len(hidden_rep)-1):
                continue
            # put pool layer here
            h_pool = global_add_pool(h, batch=batch_index)
            # h_pool = torch.sum(h, dim=0, keepdim=True)
            output_h += self.linears_prediction[layer](h_pool)

        output_h = F.relu(output_h)
        outputs = self.output_layer(output_h)

        if self.return_middle_feature:
            return outputs, output_h

        return outputs


if __name__ == '__main__':
    from torch_geometric.datasets import TUDataset
    from torch_geometric.data import DataLoader

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    data = None
    for d in loader:
        data = d
        break
    gin = GPN(2, 2, 3, 8, 5)
    outputs = gin(data.x, data.edge_index, data.batch)
    print(outputs.shape)

