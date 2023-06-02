"""
Downstream task: Tox21: Qualitative toxicity measurements on 12 biological targets, including nuclear receptors and stress response pathways
Dataset source: DeepChem
"""

import os, sys, argparse, time
import json
from tqdm import tqdm
from rdkit import Chem
import deepchem as dc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
sys.path.append('..')
from models.mol_predictor import MolPredictor
from models.series_gin_edge import SerGINE
from loader import MoleculeNetDataset
from chem import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    # environment arguments
    parser.add_argument('--cpu', default=False, action='store_true', help="train on cpu")
    parser.add_argument('--gpu', default=0, type=int, help="gpu id")
    # directory arguments
    parser.add_argument('--output_dir', default='result/Tox21', type=str, help="output directory of task")
    parser.add_argument('--model_name', default='roc_best_model.pth', type=str, help="saved model name")
    parser.add_argument('--time', default=1, type=int, help="time of experiment")
    # network arguments
    parser.add_argument('--gnn', default='SerGINE', type=str, help="GNN architecture")
    parser.add_argument('--num_atom_layers', default=3, type=int, help="num of atom-level gnn layers")
    parser.add_argument('--num_fg_layers', default=2, type=int, help="num of FG-level gnn layers")
    parser.add_argument('--emb_dim', default=128, type=int, help="embedding dimension")
    parser.add_argument('--num_tasks', default=12, type=int, help="number of tasks")
    parser.add_argument('--dropout', default=0.5, type=float, help="dropout rate")
    # training arguments
    parser.add_argument('--from_scratch', default=False, action='store_true', help="train from scratch")
    parser.add_argument('--pretrain_dir', default='../pretrained_model_cl_zinc15_250k', type=str, help="directory of pretrained models")
    parser.add_argument('--pretrain_model_name', default='model.pth', type=str, help="pretrained model name")
    parser.add_argument('--metric', default='CosineSimilarity', type=str, help="criterion of embedding distance")
    parser.add_argument('--margin', default=1.0, type=float, help="margin of contrastive loss")
    parser.add_argument('--pre_lr', default=1e-3, type=float, help="learning rate of pretraining")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr0', default=1e-3, type=float, help="learning rate of encoder")
    parser.add_argument('--lr1', default=1e-3, type=float, help="learning rate of predictor")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of training epoch")
    parser.add_argument('--log_interval', default=40, type=int, help="log interval (batch/log)")
    parser.add_argument('--early_stop', default=False, action='store_true', help="use early stop strategy")
    parser.add_argument('--patience', default=20, type=int, help="num of waiting epoch")
    parser.add_argument('--weight_decay', default=0, type=float, help="weight decay")
    parser.add_argument('--splitter', default='scaffold', choices=['scaffold', 'random'], help="Split method of dataset")

    args = parser.parse_args()

    return args


def process_data():
    tasks, datasets, transformer = dc.molnet.load_tox21(data_dir='../data/MoleculeNet', save_dir='../data/MoleculeNet',
                                                        splitter=args.splitter)
    dataset = [[], [], []]
    err_cnt = 0
    for i in range(3):
        for X, y, w, ids in datasets[i].itersamples():
            mol = Chem.MolFromSmiles(ids)
            if mol is None:
                print(f"'{ids}' cannot be convert to graph")
                err_cnt += 1
                continue
            atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list = mol_to_graphs(mol)
            if fg_features == []:  # C
                err_cnt += 1
                print(f"{ids} cannot be converted to FG graph")
                continue
            dataset[i].append([atom_features, bond_list, bond_features, fg_features, fg_edge_list, fg_edge_features, atom2fg_list, y, w])
    print(f"{err_cnt} data can't be convert to graph")
    train_set, valid_set, test_set = dataset
    return train_set, valid_set, test_set


def train(model, data_loader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    log_loss = 0
    for i, mol in enumerate(data_loader):
        mol = mol.to(device)

        optimizer.zero_grad()
        output = model(mol)
        loss = criterion(output, mol.y.reshape(-1, args.num_tasks))
        weight = mol.w.reshape(-1, args.num_tasks)
        exist_label = weight**2 > 0
        loss = loss*weight
        loss = torch.sum(loss)/torch.sum(exist_label)
        loss.backward()
        optimizer.step()
        log_loss += torch.mean(loss).item()

        # log
        if (i+1) % args.log_interval == 0:
            log_loss = log_loss/args.log_interval
            print(f"batch: {i+1}/{len(data_loader)} | loss: {log_loss :.8f} | time: {time.time()-start_time :.4f}")
            log_loss = 0


def test(model, data_loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    with torch.no_grad():
        avg_loss = 0
        true_label, pred_score = [], []
        exist_label = []
        for mol in data_loader:
            mol = mol.to(device)

            output = model(mol)
            y = mol.y.reshape(-1, args.num_tasks)
            weight = mol.w.reshape(-1, args.num_tasks)
            loss = criterion(output, y)
            exist_label_ = weight**2 > 0
            loss = loss*weight
            loss = torch.sum(loss)/torch.sum(exist_label_)
            avg_loss += torch.mean(loss).item()

            # output = sigmoid(output)
            for true, pred, exist in zip(y, output, exist_label_):
                true_label.append(true.detach().cpu().numpy())
                pred_score.append(pred.detach().cpu().numpy())
                exist_label.append(exist.detach().cpu().numpy())

        avg_loss = avg_loss/len(data_loader)
        true_label, pred_score, exist_label = np.array(true_label), np.array(pred_score), np.array(exist_label)
        auc_roc = []
        one_class_task = []
        for task in range(args.num_tasks):
            true_label_, pred_score_ = [], []
            for item in range(len(true_label)):
                if exist_label[item][task]:
                    true_label_.append(true_label[item, task])
                    pred_score_.append(pred_score[item, task])
            if sum(true_label_) == len(true_label_) or sum(true_label_) == 0:  # only one class in dataset
                one_class_task.append(task)
            else:
                auc_roc_ = roc_auc_score(true_label_, pred_score_)
                auc_roc.append(auc_roc_)
        if one_class_task != []:
            print(f"{len(one_class_task)}/{args.num_tasks} tasks have only one class, ignore roc of these tasks: {one_class_task}")
        avg_auc_roc = sum(auc_roc)/len(auc_roc)

    return avg_loss, avg_auc_roc


args = parse_args()
start_time = time.time()

output_dir = args.gnn+'_dim'+str(args.emb_dim)
output_dir = os.path.join(args.output_dir, output_dir,
                          'margin'+str(args.margin) + '_lr0_'+str(args.lr0) + '_lr1_'+str(args.lr1) + '_dropout'+str(args.dropout),
                          'time'+str(args.time))
if args.from_scratch:
    output_dir = os.path.join(output_dir, 'scratch')
ext_setting = None
if args.weight_decay > 0:
    ext_setting = 'decay'+str(args.weight_decay)
if ext_setting is not None:
    output_dir = os.path.join(output_dir, ext_setting)


def main():
    os.makedirs(output_dir, exist_ok=True)
    logger = create_file_logger(os.path.join(output_dir, 'log.txt'))
    logger.info("=======Setting=======")
    for k in args.__dict__:
        v = args.__dict__[k]
        logger.info(f"{k}: {v}")
    device = torch.device('cpu' if args.cpu else ('cuda:' + str(args.gpu)))
    logger.info(f"\nUtilized device as {device}")

    # load data
    logger.info("\n=======Process Data=======")
    train_set, valid_set, test_set = process_data()
    logger.info(f"train data num: {len(train_set)} | valid data num: {len(valid_set)} | test data num: {len(test_set)}")
    train_set = MoleculeNetDataset(train_set)
    valid_set = MoleculeNetDataset(valid_set)
    test_set = MoleculeNetDataset(test_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, follow_batch=['fg_x'])
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, follow_batch=['fg_x'])
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, follow_batch=['fg_x'])

    # define model
    if args.gnn == 'SerGINE':
        encoder = SerGINE(num_atom_layers=args.num_atom_layers, num_fg_layers=args.num_fg_layers, latent_dim=args.emb_dim,
                          atom_dim=ATOM_DIM, fg_dim=FG_DIM, bond_dim=BOND_DIM, fg_edge_dim=FG_EDGE_DIM,
                          dropout=args.dropout)
    # elif args.gnn == :  # more GNN
    else:
        raise ValueError("Undefined GNN!")
    model = MolPredictor(encoder=encoder, latent_dim=args.emb_dim, num_tasks=args.num_tasks, dropout=args.dropout)
    model.to(device)

    # load pre-trained model
    if not args.from_scratch:
        logger.info("\n=======Load Pre-trained Model=======")
        pre_path = args.gnn
        pre_path += '_dim'+str(args.emb_dim) + '_'+args.metric + '_margin'+str(args.margin) + '_lr'+str(args.pre_lr)
        pre_path = os.path.join(args.pretrain_dir, pre_path, args.pretrain_model_name)
        model.from_pretrained(model_path=pre_path, device=device)
        logger.info(f"Load pre-trained model from {pre_path}")

    logger.info("\n=======Train=======")
    os.chdir(output_dir)
    optimizer = optim.Adam([{'params': model.encoder.parameters(), 'lr': args.lr0},
                            {'params': model.predictor.parameters()}], lr=args.lr1, weight_decay=args.weight_decay)
    best = [0, 0, 0]  # [epoch, valid_roc, test_roc]
    early_stop_cnt = 0
    record = [[], [], [], [], [], []]  # [train loss, train roc, valid loss, valid roc, test loss, test roc]

    # train
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch+1 :03d}")
        early_stop_cnt += 1
        train(model, train_loader, optimizer, device)
        # record
        train_loss, train_roc = 0, 0
        # train_loss, train_roc = test(model, train_loader, device)
        record[0].append(train_loss)
        record[1].append(train_roc)
        valid_loss, valid_roc = test(model, valid_loader, device)
        record[2].append(valid_loss)
        record[3].append(valid_roc)
        test_loss, test_roc = test(model, test_loader, device)
        record[4].append(test_loss)
        record[5].append(test_roc)
        logger.info(f"Train loss: {train_loss :.8f} | AUC_ROC: {train_roc :.8f}")
        logger.info(f"Valid loss: {valid_loss :.8f} | AUC_ROC: {valid_roc :.8f}")
        logger.info(f"Test  loss: {test_loss :.8f} | AUC_ROC: {test_roc :.8f}")
        # update model
        if valid_roc > best[1]:
            best = [epoch+1, valid_roc, test_roc]
            torch.save(model.state_dict(), args.model_name)
            print(f"Saved model of Epoch {epoch+1 :03d} into '{args.model_name}'")
            early_stop_cnt = 0
        else:
            print(f"No improvement since Epoch {best[0] :03d} with Valid ROC: {best[1] :.8f} | Test ROC: {best[2] :.8f}")
        # early stop
        if args.early_stop and (early_stop_cnt == args.patience):
            logger.info(f"Early stop at Epoch {epoch+1 :03d}")
            break

    logger.info(f"\n'{args.model_name}' | Epoch: {best[0] :03d} | AUC_ROC: {best[1] :.8f}")

    logger.info("\n=======Test=======")
    logger.info(f"{args.model_name}")
    model.load_state_dict(torch.load(args.model_name, map_location=device))
    logger.info("Train set:")
    _, auc_roc = test(model, train_loader, device)
    logger.info(f"AUC_ROC: {auc_roc :.8f}")
    logger.info("Valid set:")
    _, auc_roc = test(model, valid_loader, device)
    logger.info(f"AUC_ROC: {auc_roc :.8f}")
    logger.info("Test set:")
    _, auc_roc = test(model, test_loader, device)
    logger.info(f"AUC_ROC: {auc_roc :.8f}")

    logger.info("\n=======Finish=======")


if __name__ == '__main__':
    main()
