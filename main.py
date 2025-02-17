import argparse
import torch
from utils.bpgraph_data_loader import BpGraphDataLoader
from model.bpganomodel import BPGAnomodel
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import trange


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data/', help='the root path of all dataset')
    parser.add_argument('--dataset', type=str, default='enron')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=16,
                        help='The dimension of hidden layers.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mask_prob', type=float, default=0.7)

    return parser.parse_args()


def get_auc_score(gnd, fea_gnd, fea_pre):
    residual = fea_gnd - fea_pre
    score = torch.norm(residual, p=2, dim=1).cpu()
    auc = roc_auc_score(gnd, score)

    return auc


def weighted_mse_loss(pre, gnd, weight):
    return (weight * (pre - gnd) ** 2).mean()


def main():
    args = parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    data_path = args.path + args.dataset + '.pickle'

    bpgraph_loader = BpGraphDataLoader(args.batch_size, data_path, args.dataset)
    bpgraph_loader.load(seed=args.seed, mask_prob=args.mask_prob)
    # mask = bpgraph_loader.get_mask()
    u_gnd = bpgraph_loader.get_u_gnd()
    fea_gnd = torch.Tensor(bpgraph_loader.get_fea_gnd()).cuda()
    loss_weight = torch.Tensor(bpgraph_loader.get_loss_weight()).cuda()

    bpg_model = BPGAnomodel(bpgraph_loader, args, device)
    bpg_model.cuda()
    optimiser = torch.optim.Adam(bpg_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    auc_best = 0
    best_model_path = f'./best_para/best_model_{args.dataset}.pkl'

    for epoch in range(args.epochs):
        bpg_model.train()
        optimiser.zero_grad()
        u_emb, v_emb, rating_pre = bpg_model()

        loss = weighted_mse_loss(rating_pre, fea_gnd, loss_weight)

        auc_res = get_auc_score(u_gnd, fea_gnd, rating_pre.detach())
        if auc_res > auc_best:
            auc_best = auc_res
            torch.save(bpg_model.state_dict(), best_model_path)

        print('Epoch:', (epoch + 1), '  Loss:', loss)
        if (epoch+1)%5 == 0:
            print('AUC:', auc_res)

        loss.backward()
        optimiser.step()


if __name__ == '__main__':
    main()
