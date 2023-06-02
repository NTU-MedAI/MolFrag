import os, sys, argparse, time
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
sys.path.append('..')
from models.ddi_predictor import DDIPredictor
from models.series_gin_edge import SerGINE
from loader import DDIDataset
from chem import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    # environment arguments
    parser.add_argument('--cpu', default=False, action='store_true', help="train on cpu")
    parser.add_argument('--gpu', default=0, type=int, help="gpu id")
    # directory arguments
    parser.add_argument('--dataset', default='BIOSNAP', choices=['BIOSNAP', 'DrugBankDDI', 'BIOSNAP2'], help="DDI Dataset")
    parser.add_argument('--output_dir', default='result', type=str, help="output directory")
    parser.add_argument('--model_name', default='roc_best_model.pth', help="saved model name")
    parser.add_argument('--time', default=1, type=int, help="time of experiment")
    # network arguments
    parser.add_argument('--gnn', default='SerGINE', type=str, help="GNN architecture")
    parser.add_argument('--num_atom_layers', default=3, type=int, help="num of atom-level gnn layers")
    parser.add_argument('--num_fg_layers', default=2, type=int, help="num of FG-level gnn layers")
    parser.add_argument('--emb_dim', default=128, type=int, help="embedding dimension")
    parser.add_argument('--num_tasks', default=1, type=int, help="number of tasks")
    parser.add_argument('--dropout', default=0.5, type=float, help="dropout rate")
    # training arguments
    parser.add_argument('--from_scratch', default=False, action='store_true', help="train from scratch")
    parser.add_argument('--pretrain_dir', default='../pretrained_model_cl_zinc15_250k', type=str, help="directory of pretrained models")
    parser.add_argument('--pretrain_model_name', default='model.pth', type=str, help="pretrained model name")
    parser.add_argument('--metric', default='CosineSimilarity', type=str, help="criterion of embedding distance")
    parser.add_argument('--margin', default=1.0, type=float, help="margin of contrastive loss")
    parser.add_argument('--pre_lr', default=1e-3, type=float, help="learning rate of pretraining")
    parser.add_argument('--batch_size', default=512, type=int, help="batch size")
    parser.add_argument('--lr0', default=1e-3, type=float, help="learning rate of encoder")
    parser.add_argument('--lr1', default=1e-3, type=float, help="learning rate of predictor")
    parser.add_argument('--num_epochs', default=10, type=int, help="number of training epoch")
    parser.add_argument('--log_interval', default=50, type=int, help="log interval (batch/log)")
    parser.add_argument('--early_stop', default=False, action='store_true', help="use early stop strategy")
    parser.add_argument('--patience', default=10, type=int, help="num of waiting epoch")
    parser.add_argument('--weight_decay', default=0, type=float, help="weight decay")
    parser.add_argument('--checkpoint_interval', default=10, type=int, help="checkpoint interval (epoch/checkpoint)")

    args = parser.parse_args()

    return args


def train(model, data_loader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    log_loss = 0
    for i, data in enumerate(data_loader):
        mol1, mol2, label = data
        mol1, mol2, label = mol1.to(device), mol2.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(mol1, mol2)
        loss = criterion(output, label.reshape(-1, 1))
        loss.backward()
        optimizer.step()
        log_loss += loss.item()

        # log
        if (i+1) % args.log_interval == 0:
            log_loss = log_loss/args.log_interval
            print(f"batch: {i+1}/{len(data_loader)} | loss: {log_loss :.8f} | time: {time.time()-start_time :.4f}")
            log_loss = 0


def test(model, data_loader, device, threshold: float = 0.5):
    model.eval()
    sigmoid = nn.Sigmoid()
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        avg_loss = 0
        true_label, pred_score = [], []
        for data in data_loader:
            mol1, mol2, label = data
            mol1, mol2, label = mol1.to(device), mol2.to(device), label.to(device)

            output = model(mol1, mol2)
            loss = criterion(output, label.reshape(-1, 1))
            avg_loss += loss.item()

            output = sigmoid(output)
            for true, pred in zip(label, output):
                true_label.append(true.item())
                pred_score.append(pred.item())

        avg_loss = avg_loss/len(data_loader)
        auc_roc, auc_prc, f1, mat, acc, threshold = eval_ddi_model(true_label, pred_score, threshold, opt_thr=False)
        return avg_loss, auc_roc, auc_prc, f1, mat, acc, threshold


args = parse_args()
start_time = time.time()

output_dir = args.gnn+'_dim'+str(args.emb_dim)
output_dir = os.path.join(args.output_dir, args.dataset, output_dir,
                          'margin'+str(args.margin) + '_lr0_'+str(args.lr0) + '_lr1_'+str(args.lr1) + '_dropout'+str(args.dropout),
                          'time'+str(args.time))
if args.from_scratch:
    output_dir = os.path.join(output_dir, 'scratch')
ext_setting = None
if args.weight_decay > 0:
    output_dir = os.path.join(output_dir, 'decay'+str(args.weight_decay))


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
    logger.info("\n=======Load Dataset=======")
    train_set = DDIDataset(root=os.path.join('../data/DDI', args.dataset), drug_filename='drug.csv', ddi_filename=f'train.csv')
    valid_set = DDIDataset(root=os.path.join('../data/DDI', args.dataset), drug_filename='drug.csv', ddi_filename=f'valid.csv')
    test_set = DDIDataset(root=os.path.join('../data/DDI', args.dataset), drug_filename='drug.csv', ddi_filename=f'test.csv')
    logger.info(f"train data num: {len(train_set)} | valid data num: {len(valid_set)} | test data num: {len(test_set)}")
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
    model = DDIPredictor(encoder=encoder, latent_dim=args.emb_dim, num_tasks=args.num_tasks, dropout=args.dropout)
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
    start_epoch = 0
    best = [0, 0, 0]  # [epoch, valid_roc, test_roc]
    early_stop_cnt = 0
    record = [[], [], [],  # [train roc, train prc, train f1,
              [], [], [],  #  valid_roc, valid orc, valid f1,
              [], [], []]  #  test roc, test prc, test f1]

    # load checkpoint
    if os.path.exists('checkpoint.pth'):
        checkpoint = torch.load('checkpoint.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best = checkpoint['best']
        record = checkpoint['record']
        logger.info(f"Resume training from Epoch {start_epoch+1 :03d}")

    # train
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"Epoch {epoch+1 :03d}")
        early_stop_cnt += 1
        train(model, train_loader, optimizer, device)
        # record
        train_roc, train_prc, train_f1 = 0, 0, 0
        # _, train_roc, train_prc, train_f1, _, _, _ = test(model, train_loader, device)
        record[0].append(train_roc)
        record[1].append(train_prc)
        record[2].append(train_f1)
        _, valid_roc, valid_prc, valid_f1, _, _, _ = test(model, valid_loader, device)
        record[3].append(valid_roc)
        record[4].append(valid_prc)
        record[5].append(valid_f1)
        _, test_roc, test_prc, test_f1, _, _, _ = test(model, test_loader, device)
        record[6].append(test_roc)
        record[7].append(test_prc)
        record[8].append(test_f1)
        logger.info(f"Train AUC_ROC: {train_roc :.8f} | AUC_PRC: {train_prc :.8f} | F1: {train_f1 :.8f}")
        logger.info(f"Valid AUC_ROC: {valid_roc :.8f} | AUC_PRC: {valid_prc :.8f} | F1: {valid_f1 :.8f}")
        logger.info(f"Test  AUC_ROC: {test_roc :.8f} | AUC_PRC: {test_prc :.8f} | F1: {test_f1 :.8f}")
        # update model
        if valid_roc > best[1]:
            best = [epoch+1, valid_roc, test_roc]
            torch.save(model.state_dict(), args.model_name)
            print(f"Saved model of Epoch {epoch+1 :03d} into '{args.model_name}'")
            early_stop_cnt = 0
        else:
            print(f"No improvement since Epoch {best[0] :03d} with Valid AUC_ROC: {best[1] :.8f} | Test AUC_ROC: {best[2] :.8f}")
        # save checkpoint
        if (epoch+1) % args.checkpoint_interval == 0:
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': epoch+1,
                          'best': best,
                          'record': record}
            torch.save(checkpoint, 'checkpoint.pth')
        # early stop
        if args.early_stop and (early_stop_cnt == args.patience):
            logger.info(f"Early stop at Epoch {epoch+1 :03d}")
            break

    logger.info(f"\n'{args.model_name}' | Epoch: {best[0] :03d} | AUC_ROC: {best[1] :.8f}")

    logger.info("\n=======Test=======")
    logger.info(f"{args.model_name}")
    model.load_state_dict(torch.load(args.model_name, map_location=device))
    logger.info("Train set:")
    _, auc_roc, auc_prc, f1, _, _, _ = test(model, train_loader, device)
    logger.info(f"AUC_ROC: {auc_roc :.8f} | AUC_PRC: {auc_prc :.8f} | F1: {f1 :.8f}")
    logger.info("Valid set:")
    _, auc_roc, auc_prc, f1, _, _, _ = test(model, valid_loader, device)
    logger.info(f"AUC_ROC: {auc_roc :.8f} | AUC_PRC: {auc_prc :.8f} | F1: {f1 :.8f}")
    logger.info("Test set:")
    _, auc_roc, auc_prc, f1, _, _, _ = test(model, test_loader, device)
    logger.info(f"AUC_ROC: {auc_roc :.8f} | AUC_PRC: {auc_prc :.8f} | F1: {f1 :.8f}")

    logger.info("\n=======Finish=======")


if __name__ == '__main__':
    main()
