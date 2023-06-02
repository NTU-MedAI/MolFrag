import os, json, time
from tqdm import tqdm
from collections import defaultdict
from rdkit import Chem

PATT = {
    'HETEROATOM': '[!#6]',
    'DOUBLE_TRIPLE_BOND': '*=,#*',
    'ACETAL': '[CX4]([O,N,S])[O,N,S]'
}
PATT = {k: Chem.MolFromSmarts(v) for k, v in PATT.items()}


def get_fg_set(mol):
    """
    Identify FGs and convert to SMILES
    Args:
        mol:
    Returns: a set of FG's SMILES
    """
    fgs = []  # Function Groups

    # <editor-fold desc="identify and merge rings">
    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]  # get simple rings
    flag = True  # flag == False: no rings can be merged
    while flag:
        flag = False
        for i in range(len(rings)):
            if len(rings[i]) == 0: continue
            for j in range(i + 1, len(rings)):
                shared_atoms = rings[i] & rings[j]
                if len(shared_atoms) > 2:
                    rings[i].update(rings[j])
                    rings[j] = set()
                    flag = True
    rings = [r for r in rings if len(r) > 0]
    # </editor-fold>

    # <editor-fold desc="identify functional atoms and merge connected ones">
    marks = set()
    for patt in PATT.values():  # mark functional atoms
        for sub in mol.GetSubstructMatches(patt):
            marks.update(sub)
    atom2fg = [[] for _ in range(mol.GetNumAtoms())]  # atom2fg[i]: list of i-th atom's FG idx
    for atom in marks:  # init: each marked atom is a FG
        fgs.append({atom})
        atom2fg[atom] = [len(fgs)-1]
    for bond in mol.GetBonds():  # merge FGs
        if bond.IsInRing(): continue
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in marks and a2 in marks:  # a marked atom should only belong to a FG, if atoms are both marked, merge their FGs into a FG
            assert a1 != a2
            assert len(atom2fg[a1]) == 1 and len(atom2fg[a2]) == 1
            # merge a2' FG to a1's FG
            fgs[atom2fg[a1][0]].update(fgs[atom2fg[a2][0]])
            fgs[atom2fg[a2][0]] = set()
            atom2fg[a2] = atom2fg[a1]
        elif a1 in marks:  # only one atom is marked, add neighbour atom to its FG as its environment
            assert len(atom2fg[a1]) == 1
            # add a2 to a1's FG
            fgs[atom2fg[a1][0]].add(a2)
            atom2fg[a2].extend(atom2fg[a1])
        elif a2 in marks:
            # add a1 to a2's FG
            assert len(atom2fg[a2]) == 1
            fgs[atom2fg[a2][0]].add(a1)
            atom2fg[a1].extend(atom2fg[a2])
        else:  # both atoms are unmarked, i.e. a trivial C-C single bond
            # add single bond to fgs
            fgs.append({a1, a2})
            atom2fg[a1].append(len(fgs)-1)
            atom2fg[a2].append(len(fgs)-1)

    tmp = []
    for fg in fgs:
        if len(fg) == 0: continue
        if len(fg) == 1 and mol.GetAtomWithIdx(list(fg)[0]).IsInRing(): continue
        tmp.append(fg)
    fgs = tmp
    # </editor-fold>

    fgs.extend(rings)  # final FGs: rings + FGs (not in rings)

    fg_smiles = set()
    for fg in fgs:
        fg_smiles.add(Chem.MolFragmentToSmiles(mol, fg))

    return fg_smiles


if __name__ == '__main__':
    os.chdir('data/ZINC15')

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Splitting mol to FGs...")
    with open('zinc15_250k.txt') as f:
        smiles_list = f.read().splitlines()
    print(f"# mols: {len(smiles_list)}")

    mol2fgs = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        fg_smiles = get_fg_set(mol)
        mol2fgs.append(list(fg_smiles))

    with open('mol2fgs_list.json', 'w') as f:
        json.dump(mol2fgs, f)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Generating FG corpus...")
    with open('mol2fgs_list.json', 'r') as f:
        mol2fgs = json.load(f)
    print(f"# mols: {len(mol2fgs)}")

    fg_dict = defaultdict(int)
    for fgs in tqdm(mol2fgs):
        for fg in fgs:
            fg_dict[fg] += 1
    print(f"# fgs: {len(fg_dict)}")

    fg_corpus = sorted(fg_dict.keys(), key=lambda k: fg_dict[k], reverse=True)
    fg_corpus = fg_corpus[:512]
    print(f"# corpus: {len(fg_corpus)}")
    print(f"freq: {fg_dict[fg_corpus[-1]]}/{len(mol2fgs)}")

    with open('fg_corpus.txt', 'w') as f:
        for fg in fg_corpus:
            f.write(fg+"\n")

