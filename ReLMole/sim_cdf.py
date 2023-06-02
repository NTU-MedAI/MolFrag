import os, json, time, gc
from tqdm import tqdm
import random
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

SAMP_NUM = 1000000000


if __name__ == '__main__':
    os.chdir('data/ZINC15')

    with open('zinc15_250k.txt') as f:
        smiles = f.read().splitlines()
    with open('mol2fgs_list.json', 'r') as f:
        mol2fgs = json.load(f)
    with open('fg_corpus.txt', 'r') as f:
        fg_corpus = f.read().splitlines()
    mol_num = len(smiles)
    corpus_num = len(fg_corpus)
    print(f"# mols: {mol_num}")
    print(f"# corpus: {corpus_num}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Getting FG-level vector...")
    mol2vec = np.zeros(shape=(mol_num, corpus_num), dtype=np.int8)
    for i in tqdm(range(mol_num)):
        idx = []
        for fg in mol2fgs[i]:
            try:
                idx.append(fg_corpus.index(fg))
            except:
                pass
        mol2vec[i][idx] = 1
    np.save('mol2fgvec.npy', mol2vec)

    del smiles, mol2fgs, fg_corpus, mol2vec
    gc.collect()

    with open('zinc15_250k.txt') as f:
        smiles = f.read().splitlines()
    mol_num = len(smiles)
    print(f"# mols: {mol_num}")

    # fps
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Getting FP vectors...")
    fps = []
    for smi in tqdm(smiles):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fps.append(fp)

    # fgs
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Getting FG vectors...")
    fgs = np.load('mol2fgvec.npy')
    norm = np.linalg.norm(fgs, axis=1)  # norm for calc cos sim
    norm[np.where(norm == 0)] = np.inf  # avoid div 0

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sampling mol pairs and calculating similarities...")
    fid = 0
    fp_sim, fg_sim = [], []
    for i in tqdm(range(SAMP_NUM)):
        idx1 = random.randrange(mol_num)
        idx2 = random.randrange(mol_num)
        # FP (fingerprint) Tanimoto similarity
        fp1 = fps[idx1]
        fp2 = fps[idx2]
        sim = DataStructs.FingerprintSimilarity(fp1, fp2)
        fp_sim.append(sim)
        # FG (function group) cosine similarity
        fg1 = fgs[idx1]
        fg2 = fgs[idx2]
        sim = np.dot(fg1, fg2) / (norm[idx1] * norm[idx2])
        fg_sim.append(sim)
        # save batch
        if (i+1) % 200000000 == 0 or (i+1) == SAMP_NUM:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving patch {fid}...")
            fp_sim = np.array(fp_sim, dtype=np.float16)
            np.save(f'fp_sim{fid}.npy', fp_sim)
            fg_sim = np.array(fg_sim, dtype=np.float16)
            np.save(f'fg_sim{fid}.npy', fg_sim)
            del fp_sim, fg_sim
            gc.collect()
            fp_sim, fg_sim = [], []
            fid += 1

    del smiles, fps, fgs, norm
    gc.collect()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Plotting CDF curve...")
    fp_sim = np.zeros(0, dtype=np.float16)
    for i in range(fid):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading fp_sim{i}.npy...")
        fp_sim_ = np.load(f'fp_sim{i}.npy')
        fp_sim = np.concatenate((fp_sim, fp_sim_), axis=0)
        del fp_sim_
        gc.collect()
    print(f"# mol pairs: {len(fp_sim)}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Plotting FP similarity CDF curve...")
    hist, bin_edges = np.histogram(fp_sim, bins=1000000)
    cdf = np.cumsum(hist/sum(hist))
    cdf = np.insert(cdf, 0, 0)
    plt.plot(bin_edges, cdf, label='fp_sim')
    x_major_locator = MultipleLocator(0.05)
    y_major_locator = MultipleLocator(0.05)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.grid()
    plt.xlim([-0.0001, 1.0001])
    plt.ylim([-0.0001, 1.0001])
    plt.legend(loc='lower left')
    plt.savefig('fp_sim_cdf.svg', format='svg')
    plt.clf()

    del fp_sim, hist, bin_edges, cdf
    gc.collect()

    fg_sim = np.zeros(0, dtype=np.float16)
    for i in range(fid):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading fg_sim{i}.npy...")
        fg_sim_ = np.load(f'fg_sim{i}.npy')
        fg_sim = np.concatenate((fg_sim, fg_sim_), axis=0)
        del fg_sim_
        gc.collect()
    print(f"# mol pairs: {len(fg_sim)}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Plotting FG similarity CDF curve...")
    hist, bin_edges = np.histogram(fg_sim, bins=1000000)
    cdf = np.cumsum(hist / sum(hist))
    cdf = np.insert(cdf, 0, 0)
    plt.plot(bin_edges, cdf, label='fg_sim')
    x_major_locator = MultipleLocator(0.05)
    y_major_locator = MultipleLocator(0.05)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.grid()
    plt.xlim([-0.0001, 1.0001])
    plt.ylim([-0.0001, 1.0001])
    plt.legend(loc='lower left')
    plt.savefig('fg_sim_cdf.svg', format='svg')
    plt.clf()

    del fg_sim, hist, bin_edges, cdf
    gc.collect()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finish!")

