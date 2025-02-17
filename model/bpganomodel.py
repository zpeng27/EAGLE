import numpy as np
import torch
import torch.nn as nn
from mpl import MPL, MPL_BG
import scipy.sparse as sp


class BPGAnomodel(nn.Module):
    """docstring for BPGAnomodel"""

    def __init__(self, bpgraph_loader, args, device):
        super(BPGAnomodel, self).__init__()
        self.device = device
        u_attr_dim = bpgraph_loader.get_u_attr_dim()
        v_attr_dim = bpgraph_loader.get_v_attr_dim()
        hidden_dim = args.hidden_dim

        self.batch_num_u = bpgraph_loader.get_u_batch_num()
        self.batch_num_v = bpgraph_loader.get_v_batch_num()
        self.u_attr = bpgraph_loader.get_u_attr_array()
        self.v_attr = bpgraph_loader.get_v_attr_array()
        self.u_adj = bpgraph_loader.get_u_adj()
        self.v_adj = bpgraph_loader.get_v_adj()
        self.u_adj_inner = bpgraph_loader.get_u_adj_inner()
        self.batches_u = bpgraph_loader.get_batches_u()
        self.batches_v = bpgraph_loader.get_batches_v()
        self.u_num = len(self.u_attr)
        self.v_num = len(self.v_attr)
        self.u_attr = torch.FloatTensor(self.u_attr)
        self.v_attr = torch.FloatTensor(self.v_attr)

        self.gcn_u = MPL_BG(v_attr_dim, hidden_dim, u_attr_dim)
        self.gcn_v = MPL_BG(u_attr_dim, hidden_dim, v_attr_dim)
        self.gcn_u2 = MPL_BG(hidden_dim, 8, hidden_dim)
        self.gcn_v2 = MPL_BG(hidden_dim, 8, hidden_dim)
        self.gcn_inner1 = MPL(hidden_dim, hidden_dim)
        self.gcn_inner2 = MPL(8, 8)

    def forward(self):

        v_emb_list = []
        u_attr_tensor = self.u_attr.cuda()
        for data_tuple in self.batches_v:
            v_attr_tensor = data_tuple[1].cuda()
            v_adj_tensor = data_tuple[2].cuda()

            v_emb_list.append(self.gcn_v(v_attr_tensor, u_attr_tensor, v_adj_tensor))
        v_emb = torch.cat(v_emb_list)

        u_emb_list = []
        v_attr_tensor = self.v_attr.cuda()
        for data_tuple in self.batches_u:
            u_attr_tensor = data_tuple[1].cuda()
            u_adj_tensor = data_tuple[2].cuda()

            u_emb_list.append(self.gcn_u(u_attr_tensor, v_attr_tensor, u_adj_tensor))
        u_emb = torch.cat(u_emb_list)

        u_adj_inner_tensor = self.u_adj_inner.cuda()
        u_emb = self.gcn_inner1(u_emb, u_adj_inner_tensor)
        # print ('U_EMB:', u_emb)

        ###################### 2 conv ######################

        v_emb_list = []
        for data_tuple in self.batches_v:
            v_attr_tensor = v_emb[data_tuple[0][0]:data_tuple[0][1]]
            v_adj_tensor = data_tuple[2].cuda()

            v_emb_list.append(self.gcn_v2(v_attr_tensor, u_emb, v_adj_tensor))
        v_emb2 = torch.cat(v_emb_list)

        u_emb_list = []
        for data_tuple in self.batches_u:
            u_attr_tensor = u_emb[data_tuple[0][0]:data_tuple[0][1]]
            u_adj_tensor = data_tuple[2].cuda()

            u_emb_list.append(self.gcn_u2(u_attr_tensor, v_emb, u_adj_tensor))
        u_emb2 = torch.cat(u_emb_list)

        u_emb2 = self.gcn_inner2(u_emb2, u_adj_inner_tensor)

        rating_pre = torch.tanh(torch.mm(u_emb2, torch.t(v_emb2)))

        return u_emb2, v_emb2, rating_pre
