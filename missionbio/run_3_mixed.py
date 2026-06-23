from scistreec_batch_cuda_vary_cnv import * # replaced with open-source scistreecna
import numpy as np 
import popgen  # included in our open-source package

dataset = './missionbio/3_mixed_subsample_200.npy'
reads = np.load(dataset)
n_cell, n_site, _ = reads.shape
reads = np.transpose(reads, (1, 0, 2))
reads[reads[:, :, 2] == 0] = 2
# print(reads)

gp = s2.probability.from_reads(reads[:, :, :2], cell_names=[f'{i}' for i in range(n_cell)])
caller_spr = s2.ScisTree2(threads=8, nj=True)
tree_spr, imputed_genotype_spr, likelihood_spr = caller_spr.infer(gp)
print(tree_spr)
print('scistree2', likelihood_spr)
print('start')

s = ScisTreeC(CN_MAX=5, CN_MIN=0, LAMBDA_C=100, LAMBDA_S=1, LAMBDA_T=2*n_cell-1, verbose=True)
probs = s.init_prob_leaves_gpu(reads, cnerr=0.2)
print(probs[-1])
print(reads[:, -1, :])
# tree_spr = popgen.utils.get_random_binary_tree(100)
ctree, ml = s.local_search_batch(probs, tree_spr)
print(ctree, ml)