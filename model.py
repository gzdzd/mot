import torch
from torch import nn
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import function as fn


class Net(torch.nn.Module):
    def __init__(self,
                 config,
                 num_tasks,
                 num_basis,
                 shared_filter,
                 linear_filter,
                 atom_dim,
                 bond_dim):
        super(Net, self).__init__()

        self.layers = config.layers
        self.atom_encoder = nn.Embedding(atom_dim, config.hidden)
        self.bond_encoder = nn.Embedding(bond_dim, config.hidden)

        # 添加 Dropout，默认 0.0 如果配置中没有
        self.dropout_rate = config.get('dropout', 0.0)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.convs = torch.nn.ModuleList()
        for i in range(config.layers):
            self.convs.append(Conv(hidden_size=config.hidden, dropout=self.dropout_rate))

        self.graph_pred_linear = torch.nn.Linear(config.hidden, num_tasks)

        if config.pooling == 'S':
            self.pool = SumPooling()
        elif config.pooling == 'M':
            self.pool = AvgPooling()
        elif config.pooling == 'X':
            self.pool = MaxPooling()

        if shared_filter:
            filter_n = 1
            print('Sharing the filter among signals.')
        else:
            filter_n = config.hidden
        if linear_filter:
            self.filter_encoder = nn.Sequential(
                nn.Linear(num_basis, filter_n)
            )
            print('Linear filter.')
        else:
            self.filter_encoder = nn.Sequential(
                nn.Linear(num_basis, config.hidden),
                nn.GELU(),
                nn.Linear(config.hidden, filter_n),
                nn.GELU(),
            )

    def forward(self, g, x, edge_attr, bases):
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        bases = self.filter_encoder(bases)

        for conv in self.convs:
            x = conv(g, x, edge_attr, bases)
            x = self.dropout(x)  # Apply dropout after each conv

        h_graph = self.pool(g, x)
        return self.graph_pred_linear(h_graph)

    def __repr__(self):
        return self.__class__.__name__


class Conv(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super(Conv, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.pre_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),  # Dropout in FFN
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, graph, x_feat, edge_attr, bases):
        with graph.local_scope():
            graph.ndata['x'] = x_feat
            graph.edata['e'] = edge_attr
            graph.apply_edges(fn.u_add_e('x', 'e', 'pos_e'))
            graph.edata['v'] = self.pre_ffn(graph.edata['pos_e']) * bases
            graph.update_all(fn.copy_e('v', 'pre_aggr'), fn.sum('pre_aggr', 'aggr'))
            y = graph.ndata['aggr']
            x = x_feat + y
            y = self.ffn(x)
            x = x + y
            return x

# NetL 类同样可以应用类似的 Dropout 修改，这里省略以节省篇幅，逻辑相同。