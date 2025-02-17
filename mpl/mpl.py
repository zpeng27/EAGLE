import torch
import torch.nn as nn

class MPL(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(MPL, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.Tanh() 
        self.bn = nn.BatchNorm1d(out_ft)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = self.bn(torch.spmm(adj, seq_fts))

        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class MPL_BG(nn.Module):
    def __init__(self, in_ft, out_ft, concat_ft, bias=True):
        super(MPL_BG, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.fc2 = nn.Linear((concat_ft+out_ft), out_ft, bias=False)
        self.act = nn.Tanh() 
        self.bn = nn.BatchNorm1d(out_ft)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq_self, seq, adj):
        seq_fts = self.fc(seq)
        out = self.bn(torch.spmm(adj, seq_fts))
        out = self.fc2(torch.cat((seq_self,self.act(out)),1))
        
        return self.act(out)