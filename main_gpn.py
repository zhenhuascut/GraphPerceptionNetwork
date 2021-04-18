import os
# os.environ['cuda_visible_devices'] = '0'
os.environ['cuda_visible_devices'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, separate_data
from models.GPN_model import GPN
from torch_geometric.data import Data, Batch

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    # pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    # for pos in pbar:
    for i in range(total_iters):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        data_list = [Data(g.node_features.type(torch.FloatTensor), g.edge_mat)
                     for g in batch_graph]
        batch_x = Batch.from_data_list(data_list)
        batch_x.to(device)

        output = model(batch_x.x, batch_x.edge_index, batch_x.batch)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()         
            optimizer.step()
        

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        # pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))
    
    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64, device=None):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue

        batch_graph = [graphs[idx] for idx in sampled_idx]
        data_list = [Data(g.node_features.type(torch.FloatTensor), g.edge_mat)
                     for g in batch_graph]

        batch_x = Batch.from_data_list(data_list)
        batch_x.to(device)
        model_output = model(batch_x.x, batch_x.edge_index, batch_x.batch)
        output.append(model_output.detach())

    return torch.cat(output, 0)

def model_test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs, device=device)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs, device=device)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test

def main():
    #
    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = 'MUTAG' #  'IMDBMULTI', 'COLLAB', 'NCI1' # 'REDDITBINARY' # 'MUTAG'  # 'PTC' 'PROTEINS'# 'MUTAG'
    degree_as_tag = False
    graphs, num_classes = load_data(dataset, degree_as_tag)
    print(dataset)
    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    accuracies = {}
    for fold_id in range(10):
        accuracies[fold_id] = 0
        train_graphs, test_graphs = separate_data(graphs, 0, fold_id)

        hidden_dim = 32
        final_dropout = 0.5
        learn_eps = False
        graph_pooling_type = 'sum'
        neighbor_pooling_type = 'sum' # 'attn1' 'attn2' 'sum' 'average' 'max'
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # device = torch.device('cpu')

        model = GPN(2, 2, train_graphs[0].node_features.shape[1], hidden_dim, num_classes)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.005)
        # optimizer = optim.RMSprop(model.parameters(), lr=0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        epochs = 100
        for epoch in range(1, epochs + 1):
            scheduler.step()
            from collections import namedtuple
            ARGS = namedtuple('ARGS', ['batch_size', 'iters_per_epoch'])
            args = ARGS(batch_size=128, iters_per_epoch=50)
            avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
            acc_train, acc_test = model_test(args, model, device, train_graphs, test_graphs, epoch)
            print('epoch:{} acc_train: {}, acc_test: {}'.format(epoch, acc_train, acc_test))
            # if not args.filename == "":
            #     with open(args.filename, 'w') as f:
            #         f.write("%f %f %f" % (avg_loss, acc_train, acc_test))
            #         f.write("\n")
            # print("")
            accuracies[fold_id] = max(accuracies[fold_id], acc_test)
            print('fold_id: {} current max acc test {}'.format(fold_id, accuracies[fold_id]))
            # print(model.eps)
    
    print(accuracies)
    print(np.mean(list(accuracies.values())), np.std(list(accuracies.values())))
if __name__ == '__main__':
    main()
