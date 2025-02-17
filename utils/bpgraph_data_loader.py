import pickle
import numpy as np
import networkx as nx
import scipy.sparse as sp
import math
from networkx.algorithms.bipartite import biadjacency_matrix
import torch


class BpGraphDataLoader:
    def __init__(self, batch_size, data_path, device='cpu'):

        self.device = device
        self.batch_size = batch_size
        self.data_path = data_path
        self.batch_num_u = 0
        self.batch_num_v = 0

        self.u_node_list = []
        self.u_attr_array = []

        self.v_node_list = []
        self.v_attr_array = []

        self.u_adj = []
        self.u_adj_inner = []
        self.v_adj = []

        self.u_gnd = []
        self.mask = None
        self.fea_preprocessed = None
        self.loss_weight = None

        self.batches_u = []
        self.batches_v = []

    def load(self, seed, mask_prob):
        with open(self.data_path, 'rb') as f:
            target_data = pickle.load(f)
        self.u_adj_inner = self._preprocess_adj_inner(sp.csr_matrix(target_data['A']))

        target_fea = sp.csr_matrix(target_data['X'])
        self.fea_preprocessed = self._preprocess_fea(target_fea.A)
        self.u_gnd = np.squeeze(target_data['gnd'])

        self.u_node_list = self._load_u_list(target_fea.shape[0])
        self.v_node_list = self._load_v_list(target_fea.shape[0], target_fea.shape[1])

        self.u_attr_array = self._load_u_attr(target_fea.shape[0], target_fea.shape[1])
        self.v_attr_array = self._load_v_attr(target_fea.shape[1])

        print ('U FEAT SHAPE:', self.u_attr_array.shape)
        print ('V FEAT SHAPE:', self.v_attr_array.shape)

        self.loss_weight = np.ones((target_fea.shape[0],target_fea.shape[1]))
        self.loss_weight[self.fea_preprocessed!=0] = 10  #assign weights to non-zero elements to get the expected optimization, you could adjust weights to adapt to different datasets 

        np.random.seed(seed)


        self.mask = self._gen_mask(mask_prob, target_fea.shape[0],target_fea.shape[1])
        print ('EDGE_NUM:', np.sum(self.mask), 'TRAIN_PROB:', np.sum(self.mask)/(target_fea.shape[0]*target_fea.shape[1]))
        masked_adj = self.fea_preprocessed*self.mask

        self.u_adj = sp.csr_matrix(masked_adj)
        self.v_adj = self.u_adj.transpose()

        self._gen_minibatch(self.u_attr_array, self.v_attr_array, self.u_adj, self.v_adj)

    def _build_symmetric(self, adj):
        adj = adj.tocoo()
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        return adj.tocsr()
    
    def _preprocess_adj_inner(self, adj):
        adj = adj.tocoo()
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        mx = adj + sp.eye(adj.shape[0])
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)

        return mx

    def _preprocess_fea(self, fea):  #preprocess attributes according to the dataset
        #scaler = preprocessing.MinMaxScaler()
        #scaler = preprocessing.StandardScaler()
        #fea = scaler.fit_transform(fea)
        #fea = preprocessing.normalize(fea, norm='l2', axis=0)
        return fea

    def _gen_mask(self, mask_prob, u_num, v_num):
        mask = np.random.uniform(size=(u_num,v_num))<mask_prob
        return mask


    def _load_u_list(self, u_num):
        return list(range(u_num))

    def _load_v_list(self, u_num, v_num):
        return list(range(u_num, v_num+u_num))

    def _load_u_attr(self, u_num, v_num):
        return np.ones((u_num, v_num))

    def _load_v_attr(self, v_num):
        return np.eye(v_num)

    def _load_edge_list(self):
        edges = []
        for i in self.u_node_list:
            for j in self.v_node_list:
                edges.append((i,j))
        return edges

    def _gen_adj(self, u_nodes, v_nodes, edges):
        
        bp_u = nx.Graph()
        bp_u.add_nodes_from(u_nodes, bipartite=0)
        bp_u.add_nodes_from(v_nodes, bipartite=1)
        bp_u.add_edges_from(edges)

        u_adj_np = biadjacency_matrix(bp_u, u_nodes, v_nodes)
        bp_u.clear()

        bp_v = nx.Graph()
        bp_v.add_nodes_from(v_nodes, bipartite=0)
        bp_v.add_nodes_from(u_nodes, bipartite=1)
        bp_v.add_edges_from(edges)

        v_adj_np = biadjacency_matrix(bp_v, v_nodes, u_nodes)
        bp_v.clear()

        return u_adj_np, v_adj_np


    def _gen_minibatch(self, u_attr_array, v_attr_array, u_adj, v_adj):

        u_num = u_attr_array.shape[0]
        v_num = v_attr_array.shape[0]

        self.batch_num_u = math.ceil(u_num / self.batch_size)
        self.batch_num_v = math.ceil(v_num / self.batch_size)

        for idx in range(self.batch_num_u):
            start = self.batch_size * idx
            end = self.batch_size * (idx+1)
            if idx == self.batch_num_u-1:
                end = u_num
            tmp = ((start, end), torch.FloatTensor(u_attr_array[start:end]), self._sparse_mx_to_torch_sparse_tensor(u_adj[start:end]))       
            self.batches_u.append(tmp)

        for idx in range(self.batch_num_v):
            start = self.batch_size * idx 
            end = self.batch_size * (idx+1)
            if idx == self.batch_num_v-1:
                end = v_num
            tmp = ((start, end), torch.FloatTensor(v_attr_array[start:end]), self._sparse_mx_to_torch_sparse_tensor(v_adj[start:end]))
            self.batches_v.append(tmp)
    
    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_u_adj_inner(self):
        return self._sparse_mx_to_torch_sparse_tensor(self.u_adj_inner)

    def get_u_gnd(self):
        return self.u_gnd

    def get_u_attr_dim(self):
        return len(self.u_attr_array[0])

    def get_v_attr_dim(self):
        return len(self.v_attr_array[0])

    def get_u_batch_num(self):
        return self.batch_num_u

    def get_v_batch_num(self):
        return self.batch_num_v

    def get_u_attr_array(self):
        return self.u_attr_array

    def get_v_attr_array(self):
        return self.v_attr_array

    def get_u_adj(self):
        return self.u_adj

    def get_v_adj(self):
        return self.v_adj

    def get_u_list(self):
        return self.u_node_list

    def get_v_list(self):
        return self.v_node_list

    def get_fea_gnd(self):
        return self.fea_preprocessed

    def get_mask(self):
        return self.mask

    def get_loss_weight(self):
        return self.loss_weight
    
    def get_batches_u(self):
        return self.batches_u

    def get_batches_v(self):
        return self.batches_v
