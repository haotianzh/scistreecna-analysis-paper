import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from datetime import date

import os
import numpy as np 
import pandas as pd 
import popgen
from scistree_cna import *  # can be replaced by open-source scistreecna
from copy_number_tree_builder import construct_nj_tree_with_biopython
import scistree2 as s2
from other_tools import run_cellphy_reads
from simulation_custom import simulate # simulation is included in our open-soruce package
## 
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def run_copy_num_nj(reads):
    copy_number_tree = construct_nj_tree_with_biopython(reads[:, :, 2])
    copy_number_tree = popgen.utils.from_newick(copy_number_tree)
    return copy_number_tree


def run_dice(reads):
    output_to_dice(reads)
    PATH = '/home/haz19024/miniconda3/envs/scistree2/bin/'
    os.system(f'PATH={PATH} dice -i dice_input.tsv -t -o dice_output -m balME')
    with open(f'dice_output/standard_root_balME_tree.nwk', 'r') as f:
        dice_nwk = f.readline().strip()
    dice_tree = popgen.utils.from_newick(dice_nwk)
    n_cell = len(dice_tree.get_leaves())
    dice_name_map = {f'leaf{i}': str(i) for i in range(n_cell)}
    dice_tree = popgen.utils.relabel(dice_tree, name_map=dice_name_map)
    return dice_tree


def run_cellphy(reads):
    cellphy_tree = run_cellphy_reads(reads)
    # cellphy_geno = get_cellphy_genotype('cellphy_tmp', tg)
    return cellphy_tree


def run_scistree2(reads):
    n_cell = reads.shape[1]
    gp = s2.probability.from_reads(reads[:, :, :2], cell_names=[f'{i}' for i in range(n_cell)], posterior=False)
    caller_spr = s2.ScisTree2(threads=8)
    tree_spr, imputed_genotype_spr, likelihood_spr = caller_spr.infer(gp)
    return tree_spr, imputed_genotype_spr.values


def run_scistree2_nj(reads):
    n_cell = reads.shape[1]
    gp = s2.probability.from_reads(reads[:, :, :2], cell_names=[f'{i}' for i in range(n_cell)], posterior=False)
    caller_spr = s2.ScisTree2(threads=8, nj=True)
    tree_spr, imputed_genotype_spr, likelihood_spr = caller_spr.infer(gp)
    return tree_spr, imputed_genotype_spr.values


def run_scistreec(reads, tree):
    n_site, n_cell, _ = reads.shape
    # cn_avg = estimate_copy_number(reads[:, :, -1], tree)
    s = ScisTreeC(CN_MAX=9, CN_MIN=1, LAMBDA_C=100, LAMBDA_S=1, LAMBDA_T=2*n_cell-1, verbose=False)
    probs = s.init_prob_leaves_gpu(reads, cnerr=0.05, af=0.5)
    ctree, ml = s.local_search_batch(probs, tree, tree_batch_size=8)
    ml2, indices = s.marginal_evaluate_dp(probs, ctree)
    scistreec_geno = construct_genotype(ctree, indices)
    # sites = [_ for _ in range(n_site)]
    # decoded_trees = s.viterbi_decoding(probs, ctree, sites)
    # tt = find_copy_gain_loss_on_branch(decoded_trees) 
    return ctree, scistreec_geno


def find_copy_gain_loss_on_branch(decoded_trees, gene_names=None):
    if gene_names is None:
        gene_names = [f'gene_{i}' for i in range(len(decoded_trees))]
    traversor = popgen.utils.TraversalGenerator()
    tree = decoded_trees[0].copy() # a fresh tree
    for node in traversor(tree):
        node.events = {'loss': [], 'gain': []}
    for d_tree, gene_name in zip(decoded_trees, gene_names):
        for node in traversor(d_tree):
            if node.is_root():
                if node.cn[1] != 0:
                    tree[node.name].events['gain'].append(f'{gene_name}:({node.cn[0]}: {node.cn[1]})')
            else:
                if node.cn[1] > node.parent.cn[1]:
                    tree[node.name].events['gain'].append(f'{gene_name}:({node.cn[0]}: {node.cn[1]})')
                if node.cn[1] < node.parent.cn[1] and node.cn[1] == 0:
                    tree[node.name].events['loss'].append(f'{gene_name}:({node.cn[0]}: {node.cn[1]})')
    return tree




if __name__ == "__main__":
    import popgen


    reads = np.load('HGSOC/reads_clone.npy')
    df = pd.read_csv('HGSOC/snv_cnv_subsample_200.csv')
    df_clone = pd.read_csv('HGSOC/subsample_200.csv')

    scistree2_tree, scistree2_geno = run_scistree2(reads)
    print(scistree2_tree.output())
    n_cell = reads.shape[1]
    n_site = reads.shape[0]
    reads[reads[:, :, -1] >= 5] = 5
    s = ScisTreeC(CN_MAX=5, CN_MIN=1, LAMBDA_C=100, LAMBDA_S=1, LAMBDA_T=2*n_cell-1, verbose=False)
    probs = s.init_prob_leaves_gpu(reads, cnerr=0.05, af=0.5)
    tree, ml = s.local_search_batch(probs, scistree2_tree, tree_batch_size=16)
    with open('HGSOC/scistreecna_tree_clone_max_5.txt', 'w') as out:
        out.write(tree.output())
    