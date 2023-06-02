import os, argparse, time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from utils import *
from loader import PretrainDataset
from chem import *
# model
from models.siamese_network import SiameseNetwork
from models.series_gin_edge import SerGINE


def parse_args():
    parser = argparse.ArgumentParser()
    # environment arguments
    parser.add_argument('--seed', default=0, type=int, help="set seed")
    parser.add_argument('--cpu', default=False, action='store_true', help="train on cpu")
    parser.add_argument('--gpu', default=0, type=int, help="gpu id")
    # directory arguments
    parser.add_argument('--data_dir', default='data/ZINC15', type=str, help="directory of pre-training data")
    parser.add_argument('--save_root', default='pretrained_model_cl_zinc15_250k/', type=str, help="root directory to save pretrained model")
    # network arguments
    parser.add_argument('--gnn', default='SerGINE', type=str, help="GNN architecture")
    parser.add_argument('--num_atom_layers', default=3, type=int, help="num of atom-level gnn layers")
    parser.add_argument('--num_fg_layers', default=2, type=int, help="num of FG-level gnn layers")
    parser.add_argument('--emb_dim', default=128, type=int, help="embedding dimension")
    parser.add_argument('--atom2fg_reduce', default='mean', type=str, help="atom-to-fg message passing method")
    parser.add_argument('--pool', default='mean', type=str, help="graph readout layer")
    parser.add_argument('--dropout', default=0, type=float, help="dropout rate")
    # train arguments
    parser.add_argument('--metric', default='CosineSimilarity', type=str, help="criterion of embedding distance")
    parser.add_argument('--fp_thr', default=0.22, type=float, help="threshold of fingerprint similarity")
    parser.add_argument('--fg_thr', default=0.47, type=float, help="threshold of function group similarity")
    parser.add_argument('--margin', default=1.0, type=float, help="margin of contrastive loss")
    parser.add_argument('--batch_size', default=512, type=int, help="batch size")
    parser.add_argument('--lr', default=1e-3, type=float, help="learning rate")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of training epoch")
    parser.add_argument('--log_interval', default=100, type=int, help="log interval (batch/log)")
    parser.add_argument('--checkpoint_interval', default=10, type=int, help="checkpoint interval (epoch/checkpoint)")

    args = parser.parse_args()
    return args


def train(model, data_loader, optimizer, device):
    model.train()
    criterion = ContrastiveLoss(metric=args.metric, neg_margin=args.margin, reduction='AvgNonZero', return_distance=False)
    log_loss, avg_loss = 0, 0
    for i, mol in enumerate(data_loader):
        mol = mol.to(device)

        optimizer.zero_grad()
        emb = model(mol)

        # Tanimoto similarity of FP
        inter = torch.mm(mol.fp, mol.fp.t())
        bit_cnt = torch.sum(mol.fp, dim=1, keepdim=True)
        union = bit_cnt + bit_cnt.t() - inter
        fp_sim = inter / union
        # cosine similarity of FG
        fg_sim = F.cosine_similarity(mol.fg.unsqueeze(1), mol.fg.unsqueeze(0), dim=2)
        # label
        label = (fp_sim > args.fp_thr) * (fg_sim > args.fg_thr)

        # contrastive loss
        loss = criterion(emb, label.float())
        loss.backward()
        optimizer.step()
        log_loss += loss.item()
        avg_loss += loss.item()

        # log
        if (i+1) % args.log_interval == 0:
            log_loss = log_loss / args.log_interval
            print(f"batch: {i+1}/{len(data_loader)} | loss: {log_loss :.8f} | time: {time.time()-start_time :.4f}")
            log_loss = 0
    avg_loss = avg_loss / len(data_loader)
    return avg_loss


def valid(model, data_loader, device):
    model.eval()
    criterion = ContrastiveLoss(metric=args.metric, neg_margin=args.margin, reduction='AvgNonZero', return_distance=False)
    with torch.no_grad():
        avg_loss = 0
        for mol in data_loader:
            mol = mol.to(device)

            emb = model(mol)

            # Tanimoto similarity of FP
            inter = torch.mm(mol.fp, mol.fp.t())
            bit_cnt = torch.sum(mol.fp, dim=1, keepdim=True)
            union = bit_cnt + bit_cnt.t() - inter
            fp_sim = inter / union
            # cosine similarity of FG
            fg_sim = F.cosine_similarity(mol.fg.unsqueeze(1), mol.fg.unsqueeze(0), dim=2)
            # label
            label = (fp_sim > args.fp_thr) * (fg_sim > args.fg_thr)

            loss = criterion(emb, label.float())
            avg_loss += loss.item()

        avg_loss = avg_loss/len(data_loader)

    return avg_loss


args = parse_args()
start_time = time.time()
model_save_dir = os.path.join(args.save_root,
                              args.gnn +
                              '_dim'+str(args.emb_dim) +
                              '_'+args.metric +
                              '_margin'+str(args.margin) +
                              '_lr'+str(args.lr))


def main():
    set_seed(args.seed)
    os.makedirs(model_save_dir, exist_ok=True)
    logger = create_file_logger(os.path.join(model_save_dir, 'log.txt'))
    logger.info(f"\n\n======={time.strftime('%Y-%m-%d %H:%M:%S')}=======\n")
    logger.info("=======Setting=======")
    for k in args.__dict__:
        v = args.__dict__[k]
        logger.info(f"{k}: {v}")

    # load data
    if not os.path.exists(args.data_dir):
        print("Data directory not found!")
        return
    train_set = PretrainDataset(root=args.data_dir,
                                mol_filename='zinc15_250k.txt',
                                fg_corpus_filename='fg_corpus.txt',
                                mol2fgs_filename='mol2fgs_list.json')
    logger.info(f"train data num: {len(train_set)}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, follow_batch=['fg_x'])

    logger.info("\n=======Pre-train Start=======")
    device = torch.device('cpu' if args.cpu else ('cuda:' + str(args.gpu)))
    logger.info(f"Utilized device as {device}")
    os.chdir(model_save_dir)
    start_epoch = 0
    loss_record = []  # train loss

    # define model and optimizer
    if args.gnn == 'SerGINE':
        encoder = SerGINE(num_atom_layers=args.num_atom_layers, num_fg_layers=args.num_fg_layers,
                          latent_dim=args.emb_dim, atom_dim=ATOM_DIM, fg_dim=FG_DIM,
                          bond_dim=BOND_DIM, fg_edge_dim=FG_EDGE_DIM,
                          atom2fg_reduce=args.atom2fg_reduce, pool=args.pool,
                          dropout=args.dropout)
    # elif args.gnn == :
    else:
        raise ValueError("Undefined GNN!")
    model = SiameseNetwork(encoder=encoder, latent_dim=args.emb_dim, dropout=args.dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # load checkpoint
    if os.path.exists('checkpoint.pth'):
        checkpoint = torch.load('checkpoint.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_record = checkpoint['loss_record']
        logger.info(f"Resume pre-training from Epoch {start_epoch+1 :03d}")

    # train
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"Epoch {epoch+1 :03d}")
        train_loss = train(model, train_loader, optimizer, device)
        loss_record.append(train_loss)
        logger.info(f"Train loss: {train_loss :.8f}")  # | Valid loss: {valid_loss :.8f}")
        torch.save(model.encoder.state_dict(), 'model.pth')
        # save checkpoint
        if (epoch+1) % args.checkpoint_interval == 0:
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epoch': epoch+1,
                          'loss_record': loss_record}
            torch.save(checkpoint, 'checkpoint.pth')

    logger.info("=======Pre-train Finish=======")



if __name__ == "__main__":
    main()
