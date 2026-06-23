import os
import numpy as np 
import pandas as pd 
import popgen
import pickle
# from scistreec_batch_cuda_vary_cnv import *
from scistree_cna import *
from copy_number_tree_builder import construct_nj_tree_with_biopython
import scistree2 as s2
from other_tools import run_cellphy_reads
from simulation_custom import simulate


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
    cellphy_geno = get_cellphy_genotype('cellphy_tmp', tg)
    return cellphy_tree, cellphy_geno


def run_scistree2(reads):
    n_cell = reads.shape[1]
    gp = s2.probability.from_reads(reads[:, :, :2], cell_names=[f'{i}' for i in range(n_cell)], posterior=False)
    caller_spr = s2.ScisTree2(threads=8)
    tree_spr, imputed_genotype_spr, likelihood_spr = caller_spr.infer(gp)
    return tree_spr, imputed_genotype_spr.values


def output_to_condor(reads, tree):
    df_variant = pd.DataFrame(reads[:, :, 1]).T
    df_total = pd.DataFrame(reads[:, :, 0] + reads[:, :, 1]).T
    df_character_matrix = pd.DataFrame(np.zeros((reads.shape[1], reads.shape[0]), dtype=int))
    cluster_ids = []
    for i in range(reads.shape[1]):
        cell_name = str(i)
        cluster_ids.append(tree[cell_name].cid)
    df_character_matrix['cluster_id'] = cluster_ids
    prefix = 'condor_input'
    df_variant.to_csv(f'{prefix}_variant.csv')
    df_total.to_csv(f'{prefix}_total.csv')
    df_character_matrix.to_csv(f'{prefix}_charater_matrix.csv')


def run_condor(reads, tree):
    PATH = '/data/haotian/snvcnv/condor/ConDoR/src/condor.py'
    prefix = 'condor_input'
    output_to_condor(reads, tree)
    os.system(f'python {PATH} -i {prefix}_charater_matrix.csv -a 0.002 -b 0.001 -k 3 -r {prefix}_total.csv -v {prefix}_variant.csv -o {prefix}')
    with open(f'{prefix}_tree.newick', 'r') as f:
        tree = f.readline().strip()
    condor_tree = popgen.utils.from_newick(tree)
    df = pd.read_csv(f'{prefix}_B.csv').iloc[:, 1:]
    condor_geno = df.T.values
    return condor_tree, condor_geno


def run_scistreec(reads, tree):
    n_cell = reads.shape[1]
    cn_avg = estimate_copy_number(reads[:, :, -1], tree)
    s = ScisTreeC(CN_MAX=5, CN_MIN=1, LAMBDA_C=cn_avg, LAMBDA_S=1, LAMBDA_T=2*n_cell-1, verbose=False)
    probs = s.init_prob_leaves_gpu(reads, cnerr=0.05, af=0.5)
    ctree, ml = s.local_search_batch(probs, tree)
    ml2, indices = s.marginal_evaluate_dp(probs, ctree)
    scistreec_geno = construct_genotype(ctree, indices)
    return ctree, scistreec_geno


def add_noise(reads, cnerr=0.05):
    reads, masks = add_copy_number_noise2(reads, noise_prob=cnerr) # add noise to copy numbers
    reads[reads[:, :, 2] < 0] = 0
    reads[reads[:, :, 2] > 5] = 5
    return reads


def add_mask(reads, maskerr=0.05):
    reads, masks = add_copy_number_random_mask(reads, noise_prob=maskerr) # add noise to copy numbers
    reads[reads[:, :, 2] > 5] = 5
    return reads


def add_copy_number_random_mask(reads, mask_prob):
    noisy_matrix = reads.copy()
    copy_numbers = noisy_matrix[:, :, 2]
    flip_mask = np.random.rand(*copy_numbers.shape) < mask_prob
    noisy_matrix[:, :, 2][flip_mask] = -1
    return noisy_matrix, flip_mask



# run cellcoal simulation
def run_simulation_cellcoal():
    results = []
    dirs = ['./simulation/test_with_cn_ado_50_50_less_mut_more_cn',
            './simulation/test_with_cn_ado_100_100_less_mut_more_cn', 
            './simulation/test_with_cn_ado_150_150_less_mut_more_cn']
    dirs = ['./simulation/test_with_cn_ado_200_200_less_mut_more_cn']
    for d in dirs:
        for i in range(1, 11):
            try:
                reads, tree , tg_gt = get_scistreec_input_with_cn(d, i)
                n_site, n_cell, _ = reads.shape
                rename_dict = {f'cell{i+1:04}': str(i+1) for i in range(n_cell)}
                tree = popgen.utils.relabel(tree, name_map=rename_dict)
                tg = load_true_genotype(d, i)
                nwk = tree.output()
                tree = get_true_tree(nwk)
                reads = add_noise(reads)
        
                copy_number_tree = run_copy_num_nj(reads)
                dice_tree = run_dice(reads)
                scistree2_tree, scistree2_geno = run_scistree2(reads)
                cellphy_tree = run_cellphy_reads(reads)
                cellphy_geno = get_cellphy_genotype('cellphy_tmp', tg)
                scistreec_tree, scistreec_geno = run_scistreec(reads, scistree2_tree)

                copy_number_tree_acc = tree_accuracy(tree, copy_number_tree)
                dice_tree_acc = tree_accuracy(tree, dice_tree)
                scistree2_tree_acc = tree_accuracy(tree, scistree2_tree)
                cellphy_tree_acc = tree_accuracy(tree, cellphy_tree)
                scistreec_tree_acc = tree_accuracy(tree, scistreec_tree)

                scistree2_geno_acc = genotype_accuarcy(tg, scistree2_geno)
                cellphy_geno_acc = genotype_accuarcy(tg, cellphy_geno)
                scistreec_geno_acc = genotype_accuarcy(tg, scistreec_geno)


                print("============ Tree Acc ==============")
                print('copy NJ tree acc:', copy_number_tree_acc)
                print('dice tree acc:', dice_tree_acc)
                print('scistree2 tree acc:', scistree2_tree_acc)
                print('cellphy tree acc:', cellphy_tree_acc)
                print('scistreec tree acc:', scistreec_tree_acc)
                print("----------- Genotype Acc -----------")
                print('scistree2 geno acc:', scistree2_geno_acc)
                print('cellphy geno acc:', cellphy_geno_acc)
                print('scistreec geno acc:', scistreec_geno_acc)
                print("====================================")

                results.extend([[d, i, 'copy_nj', 'tree_acc', copy_number_tree_acc],
                                [d, i, 'dice', 'tree_acc', dice_tree_acc],
                                [d, i, 'scistree2', 'tree_acc', scistree2_tree_acc], 
                                [d, i, 'cellphy', 'tree_acc', cellphy_tree_acc],
                                [d, i, 'scistreec', 'tree_acc', scistreec_tree_acc],
                                [d, i, 'scistree2', 'geno_acc', scistree2_geno_acc],
                                [d, i, 'cellphy', 'geno_acc', cellphy_geno_acc],
                                [d, i, 'scistreec', 'geno_acc', scistreec_geno_acc]])
            except Exception as e:
                pass
    df = pd.DataFrame(results, columns=['dataset', 'index', 'method', 'metric', 'value'])
    df.to_csv('cellcoal_simulation_results_cell200.csv')
    print(df)



def generate_data_scism_clt(n_cell, n_site, coverage_mean=10, coverage_std=5):
    tree = popgen.utils.get_random_binary_tree(n_cell)
    tree.root.branch = 1
    popgen.utils.apply_attr_on_tree(tree, 'branch', lambda x: x.branch / 5)
    reads, tg= simulate.generate_sample(tree, n_site, coverage_mean, coverage_std)
    tg = ggeno_to_bgeno(tg)
    return reads, tg, tree



def generate_data_scsim_clone(n_cell, n_site, n_cluster):
    tree = simulate.generate_tree(n_cells=n_cell, n_clusters=n_cluster)
    reads, tg= simulate.generate_sample2(tree, n_site)
    tg = ggeno_to_bgeno(tg)
    return reads, tg, tree



def save(dirname, name, data):
    reads, tg, tree = data
    np.save(f'{dirname}/{name}.reads.npy', reads)
    np.save(f'{dirname}/{name}.tg.npy', tg)
    with open(f'{dirname}/{name}.tree.txt', 'w') as out:
        out.write(tree.output(branch_length_func=lambda x: x.branch))
    with open(f'{dirname}/{name}.tree.pkl', 'wb') as out:
        pickle.dump(tree, out)


def run_simulation_scsim_clt():
    # different settings, simulate data and save it at the same time 
    dirname = 'simulation_scsim_clt'
    n_repeats = 20
    n_cells = [50, 100, 150, 200]
    results = []
    for n_cell in n_cells:
        n_site = n_cell
        if n_cell == 200:
            n_repeats = 10
        for i in range(n_repeats):
            try:
                data = generate_data_scism_clt(n_cell, n_site)
                save(dirname=dirname, name=f'cell{n_cell}_site{n_site}_{i}', data=data)
                reads, tg, tree = data
                reads = add_noise(reads)

                copy_number_tree = run_copy_num_nj(reads)
                dice_tree = run_dice(reads)
                scistree2_tree, scistree2_geno = run_scistree2(reads)
                cellphy_tree = run_cellphy_reads(reads)
                cellphy_geno = get_cellphy_genotype('cellphy_tmp', tg)
                scistreec_tree, scistreec_geno = run_scistreec(reads, scistree2_tree)
                with open(f'{dirname}/cell{n_cell}_site{n_site}_{i}.scistreec.txt', 'w') as out:
                    out.write(scistree2_tree.output())

                copy_number_tree_acc = tree_accuracy(tree, copy_number_tree)
                dice_tree_acc = tree_accuracy(tree, dice_tree)
                scistree2_tree_acc = tree_accuracy(tree, scistree2_tree)
                cellphy_tree_acc = tree_accuracy(tree, cellphy_tree)
                scistreec_tree_acc = tree_accuracy(tree, scistreec_tree)

                scistree2_geno_acc = genotype_accuarcy(tg, scistree2_geno)
                cellphy_geno_acc = genotype_accuarcy(tg, cellphy_geno)
                scistreec_geno_acc = genotype_accuarcy(tg, scistreec_geno)


                print("============ Tree Acc ==============")
                print('copy NJ tree acc:', copy_number_tree_acc)
                print('dice tree acc:', dice_tree_acc)
                print('scistree2 tree acc:', scistree2_tree_acc)
                print('cellphy tree acc:', cellphy_tree_acc)
                print('scistreec tree acc:', scistreec_tree_acc)
                print("----------- Genotype Acc -----------")
                print('scistree2 geno acc:', scistree2_geno_acc)
                print('cellphy geno acc:', cellphy_geno_acc)
                print('scistreec geno acc:', scistreec_geno_acc)
                print("====================================")

                results.extend([[f'{n_cell}_{n_site}', i, 'copy_nj', 'tree_acc', copy_number_tree_acc],
                                [f'{n_cell}_{n_site}', i, 'dice', 'tree_acc', dice_tree_acc],
                                [f'{n_cell}_{n_site}', i, 'scistree2', 'tree_acc', scistree2_tree_acc], 
                                [f'{n_cell}_{n_site}', i, 'cellphy', 'tree_acc', cellphy_tree_acc],
                                [f'{n_cell}_{n_site}', i, 'scistreec', 'tree_acc', scistreec_tree_acc],
                                [f'{n_cell}_{n_site}', i, 'scistree2', 'geno_acc', scistree2_geno_acc],
                                [f'{n_cell}_{n_site}', i, 'cellphy', 'geno_acc', cellphy_geno_acc],
                                [f'{n_cell}_{n_site}', i, 'scistreec', 'geno_acc', scistreec_geno_acc]])
            except Exception as e:
                pass
            print(f'complete {n_cell}_{n_site}_{i}')
    df = pd.DataFrame(results, columns=['dataset', 'index', 'method', 'metric', 'value'])
    df.to_csv('scsim_simulation_results.csv', index=None)



def run_simulation_scsim_clt_low_coverage():
    dirname = 'simulation_scsim_clt_low_coverage_high_cn_noise_500_low'
    os.makedirs(dirname, exist_ok=True)
    n_repeats = 5
    # n_cells = [50, 100, 150, 200]
    n_cells = [500]
    results = []
    for n_cell in n_cells:
        n_site = n_cell
        # if n_cell == 200:
        #     n_repeats = 5
        for i in range(n_repeats):
            try:
                data = generate_data_scism_clt(n_cell, n_site, coverage_mean=1, coverage_std=1)
                save(dirname=dirname, name=f'cell{n_cell}_site{n_site}_{i}', data=data)
                reads, tg, tree = data
                # reads = add_noise(reads, cnerr=0.1)
                reads = random_mask_missing(reads, missing_prob=0.2)
                rr = reads.copy()
                rr[rr < 1] = 1
                dice_tree = run_dice(reads)
                copy_number_tree = run_copy_num_nj(reads)
                
                scistree2_tree, scistree2_geno = run_scistree2(reads)
                cellphy_tree = run_cellphy_reads(reads)
                cellphy_geno = get_cellphy_genotype('cellphy_tmp', tg)
                # scistreec_tree, scistreec_geno = run_scistreec(reads, scistree2_tree)
                # with open(f'{dirname}/cell{n_cell}_site{n_site}_{i}.scistreec.txt', 'w') as out:
                #     out.write(scistree2_tree.output())

                copy_number_tree_acc = tree_accuracy(tree, copy_number_tree)
                dice_tree_acc = tree_accuracy(tree, dice_tree)
                scistree2_tree_acc = tree_accuracy(tree, scistree2_tree)
                cellphy_tree_acc = tree_accuracy(tree, cellphy_tree)
                # scistreec_tree_acc = tree_accuracy(tree, scistreec_tree)

                scistree2_geno_acc = genotype_accuarcy(tg, scistree2_geno)
                cellphy_geno_acc = genotype_accuarcy(tg, cellphy_geno)
                # scistreec_geno_acc = genotype_accuarcy(tg, scistreec_geno)


                print("============ Tree Acc ==============")
                print('copy NJ tree acc:', copy_number_tree_acc)
                print('dice tree acc:', dice_tree_acc)
                print('scistree2 tree acc:', scistree2_tree_acc)
                print('cellphy tree acc:', cellphy_tree_acc)
                # print('scistreec tree acc:', scistreec_tree_acc)
                print("----------- Genotype Acc -----------")
                print('scistree2 geno acc:', scistree2_geno_acc)
                print('cellphy geno acc:', cellphy_geno_acc)
                # print('scistreec geno acc:', scistreec_geno_acc)
                print("====================================")

                results.extend([[f'{n_cell}_{n_site}', i, 'copy_nj', 'tree_acc', copy_number_tree_acc],
                                [f'{n_cell}_{n_site}', i, 'dice', 'tree_acc', dice_tree_acc],
                                [f'{n_cell}_{n_site}', i, 'scistree2', 'tree_acc', scistree2_tree_acc], 
                                [f'{n_cell}_{n_site}', i, 'cellphy', 'tree_acc', cellphy_tree_acc],
                                # [f'{n_cell}_{n_site}', i, 'scistreec', 'tree_acc', scistreec_tree_acc],
                                [f'{n_cell}_{n_site}', i, 'scistree2', 'geno_acc', scistree2_geno_acc],
                                [f'{n_cell}_{n_site}', i, 'cellphy', 'geno_acc', cellphy_geno_acc],
                                # [f'{n_cell}_{n_site}', i, 'scistreec', 'geno_acc', scistreec_geno_acc]]
                                ])
            except Exception as e:
                pass
            print(f'complete {n_cell}_{n_site}_{i}')
    df = pd.DataFrame(results, columns=['dataset', 'index', 'method', 'metric', 'value'])
    df.to_csv('scsim_simulation_results_low_coverage_high_cn_noise_n500_low.csv', index=None)




def run_simulation_scsim_clone():
    # different settings, simulate data and save it at the same time 
    dirname = 'simulation_scsim_clone_2'
    # n_cells = [50, 100, 150]
    n_cells = [50]
    n_clusters = [4, 6]
    n_repeats = 5
    results = []
    for n_cluster in n_clusters:
        for n_cell in n_cells:
            n_site = n_cell
            for i in range(n_repeats):
                try:
                    data = generate_data_scsim_clone(n_cell, n_site, n_cluster)
                    save(dirname=dirname, name=f'cell{n_cell}_cluster{n_cluster}_{i}', data=data)
                    reads, tg, tree = data
                    # reads = add_noise(reads)

                    copy_number_tree = run_copy_num_nj(reads)
                    dice_tree = run_dice(reads)
                    scistree2_tree, scistree2_geno = run_scistree2(reads)
                    cellphy_tree = run_cellphy_reads(reads)
                    cellphy_geno = get_cellphy_genotype('cellphy_tmp', tg)
                    scistreec_tree, scistreec_geno = run_scistreec(reads, scistree2_tree)
                    with open(f'{dirname}/cell{n_cell}_site{n_site}_{i}.scistreec.txt', 'w') as out:
                        out.write(scistreec_tree.output())

                    condor_tree, condor_geno = run_condor(reads, tree)

                    copy_number_tree_acc = tree_accuracy(tree, copy_number_tree)
                    dice_tree_acc = tree_accuracy(tree, dice_tree)
                    scistree2_tree_acc = tree_accuracy(tree, scistree2_tree)
                    cellphy_tree_acc = tree_accuracy(tree, cellphy_tree)
                    scistreec_tree_acc = tree_accuracy(tree, scistreec_tree)

                    scistree2_geno_acc = genotype_accuarcy(tg, scistree2_geno)
                    cellphy_geno_acc = genotype_accuarcy(tg, cellphy_geno)
                    scistreec_geno_acc = genotype_accuarcy(tg, scistreec_geno)

                    condor_tree_acc = tree_accuracy(tree, condor_tree)
                    condor_geno_acc = genotype_accuarcy(tg, condor_geno)


                    print("============ Tree Acc ==============")
                    print('copy NJ tree acc:', copy_number_tree_acc)
                    print('dice tree acc:', dice_tree_acc)
                    print('scistree2 tree acc:', scistree2_tree_acc)
                    print('cellphy tree acc:', cellphy_tree_acc)
                    print('scistreec tree acc:', scistreec_tree_acc)
                    print('condor tree acc:', condor_tree_acc)
                    print("----------- Genotype Acc -----------")
                    print('scistree2 geno acc:', scistree2_geno_acc)
                    print('cellphy geno acc:', cellphy_geno_acc)
                    print('scistreec geno acc:', scistreec_geno_acc)
                    print('condor geno acc:', condor_geno_acc)
                    print("====================================")

                    results.extend([[f'{n_cell}_{n_cluster}', i, 'copy_nj', 'tree_acc', copy_number_tree_acc],
                                    [f'{n_cell}_{n_cluster}', i, 'dice', 'tree_acc', dice_tree_acc],
                                    [f'{n_cell}_{n_cluster}', i, 'scistree2', 'tree_acc', scistree2_tree_acc], 
                                    [f'{n_cell}_{n_cluster}', i, 'cellphy', 'tree_acc', cellphy_tree_acc],
                                    [f'{n_cell}_{n_cluster}', i, 'scistreec', 'tree_acc', scistreec_tree_acc],
                                    [f'{n_cell}_{n_cluster}', i, 'condor', 'tree_acc', condor_tree_acc],
                                    [f'{n_cell}_{n_cluster}', i, 'scistree2', 'geno_acc', scistree2_geno_acc],
                                    [f'{n_cell}_{n_cluster}', i, 'cellphy', 'geno_acc', cellphy_geno_acc],
                                    [f'{n_cell}_{n_cluster}', i, 'scistreec', 'geno_acc', scistreec_geno_acc],
                                    [f'{n_cell}_{n_cluster}', i, 'condor', 'geno_acc', condor_geno_acc],
                                    ])
                except Exception as e:
                    print(e)
                print(f'complete {n_cluster}_{n_cell}_{i}')
        #         break
        #     break
        # break
    df = pd.DataFrame(results, columns=['dataset', 'index', 'method', 'metric', 'value'])
    df.to_csv('scsim_simulation_clone_condor_50_2.csv', index=None)


def run_simulation_noise():
    dirname = 'simulation_scsim_noise'
    n_cell = 100
    n_site = 100
    n_repeats = 10
    noises = [0.15]
    results = []
    for noise in noises:
        for i in range(n_repeats):
            try:
                data = generate_data_scism_clt(n_cell, n_site)
                save(dirname=dirname, name=f'cell{n_cell}_site{n_site}_{i}', data=data)
                reads, tg, tree = data
                reads = add_noise(reads, cnerr=noise)

                copy_number_tree = run_copy_num_nj(reads)
                dice_tree = run_dice(reads)
                scistree2_tree, scistree2_geno = run_scistree2(reads)
                cellphy_tree = run_cellphy_reads(reads)
                cellphy_geno = get_cellphy_genotype('cellphy_tmp', tg)
                scistreec_tree, scistreec_geno = run_scistreec(reads, scistree2_tree)
                with open(f'{dirname}/cell{n_cell}_site{n_site}_{i}.scistreec.txt', 'w') as out:
                    out.write(scistree2_tree.output())

                copy_number_tree_acc = tree_accuracy(tree, copy_number_tree)
                dice_tree_acc = tree_accuracy(tree, dice_tree)
                scistree2_tree_acc = tree_accuracy(tree, scistree2_tree)
                cellphy_tree_acc = tree_accuracy(tree, cellphy_tree)
                scistreec_tree_acc = tree_accuracy(tree, scistreec_tree)

                scistree2_geno_acc = genotype_accuarcy(tg, scistree2_geno)
                cellphy_geno_acc = genotype_accuarcy(tg, cellphy_geno)
                scistreec_geno_acc = genotype_accuarcy(tg, scistreec_geno)


                print("============ Tree Acc ==============")
                print('copy NJ tree acc:', copy_number_tree_acc)
                print('dice tree acc:', dice_tree_acc)
                print('scistree2 tree acc:', scistree2_tree_acc)
                print('cellphy tree acc:', cellphy_tree_acc)
                print('scistreec tree acc:', scistreec_tree_acc)
                print("----------- Genotype Acc -----------")
                print('scistree2 geno acc:', scistree2_geno_acc)
                print('cellphy geno acc:', cellphy_geno_acc)
                print('scistreec geno acc:', scistreec_geno_acc)
                print("====================================")

                results.extend([[noise, i, 'copy_nj', 'tree_acc', copy_number_tree_acc],
                                [noise, i, 'dice', 'tree_acc', dice_tree_acc],
                                [noise, i, 'scistree2', 'tree_acc', scistree2_tree_acc], 
                                [noise, i, 'cellphy', 'tree_acc', cellphy_tree_acc],
                                [noise, i, 'scistreec', 'tree_acc', scistreec_tree_acc],
                                [noise, i, 'scistree2', 'geno_acc', scistree2_geno_acc],
                                [noise, i, 'cellphy', 'geno_acc', cellphy_geno_acc],
                                [noise, i, 'scistreec', 'geno_acc', scistreec_geno_acc]])
            except Exception as e:
                pass
            print(f'complete {noise}_{i}')
        #     break
        # break
    df = pd.DataFrame(results, columns=['dataset', 'index', 'method', 'metric', 'value'])
    df.to_csv('scsim_noise2.csv', index=None)



def run_simulation_missing():
    dirname = 'simulation_scsim_missing'
    n_cell = 100
    n_site = 100
    n_repeats = 10
    missing_rates = [0.1, 0.3, 0.5, 0.7]
    results = []
    for noise in missing_rates:
        for i in range(n_repeats):
            try:
                data = generate_data_scism_clt(n_cell, n_site)
                save(dirname=dirname, name=f'cell{n_cell}_site{n_site}_{i}', data=data)
                reads, tg, tree = data
                reads = add_noise(reads)

                reads = random_mask_missing(reads, missing_prob=noise)

                copy_number_tree = run_copy_num_nj(reads)
                dice_tree = run_dice(reads)
                scistree2_tree, scistree2_geno = run_scistree2(reads)
                cellphy_tree = run_cellphy_reads(reads)
                cellphy_geno = get_cellphy_genotype('cellphy_tmp', tg)
                scistreec_tree, scistreec_geno = run_scistreec(reads, scistree2_tree)
                with open(f'{dirname}/cell{n_cell}_site{n_site}_{i}.scistreec.txt', 'w') as out:
                    out.write(scistreec_tree.output())

                copy_number_tree_acc = tree_accuracy(tree, copy_number_tree)
                dice_tree_acc = tree_accuracy(tree, dice_tree)
                scistree2_tree_acc = tree_accuracy(tree, scistree2_tree)
                cellphy_tree_acc = tree_accuracy(tree, cellphy_tree)
                scistreec_tree_acc = tree_accuracy(tree, scistreec_tree)

                scistree2_geno_acc = genotype_accuarcy(tg, scistree2_geno)
                cellphy_geno_acc = genotype_accuarcy(tg, cellphy_geno)
                scistreec_geno_acc = genotype_accuarcy(tg, scistreec_geno)


                print("============ Tree Acc ==============")
                print('copy NJ tree acc:', copy_number_tree_acc)
                print('dice tree acc:', dice_tree_acc)
                print('scistree2 tree acc:', scistree2_tree_acc)
                print('cellphy tree acc:', cellphy_tree_acc)
                print('scistreec tree acc:', scistreec_tree_acc)
                print("----------- Genotype Acc -----------")
                print('scistree2 geno acc:', scistree2_geno_acc)
                print('cellphy geno acc:', cellphy_geno_acc)
                print('scistreec geno acc:', scistreec_geno_acc)
                print("====================================")

                results.extend([[noise, i, 'copy_nj', 'tree_acc', copy_number_tree_acc],
                                [noise, i, 'dice', 'tree_acc', dice_tree_acc],
                                [noise, i, 'scistree2', 'tree_acc', scistree2_tree_acc], 
                                [noise, i, 'cellphy', 'tree_acc', cellphy_tree_acc],
                                [noise, i, 'scistreec', 'tree_acc', scistreec_tree_acc],
                                [noise, i, 'scistree2', 'geno_acc', scistree2_geno_acc],
                                [noise, i, 'cellphy', 'geno_acc', cellphy_geno_acc],
                                [noise, i, 'scistreec', 'geno_acc', scistreec_geno_acc]])
            except Exception as e:
                pass
            print(f'complete {noise}_{i}')
        #     break
        # break
    df = pd.DataFrame(results, columns=['dataset', 'index', 'method', 'metric', 'value'])
    df.to_csv('scsim_missing.csv', index=None)


def run_simulation_homo():
    pass





def run_simulation_runtime():
    from time import time
    # n_cells = [50, 100, 150]
    # n_sites = [50, 200, 500]
    n_cells = [500]
    n_sites = [50, 200, 500]
    n_repeats = 4
    results = []
    for n_cell in n_cells:
        for n_site in n_sites:
            for i in range(n_repeats):
                try:
                    data = generate_data_scism_clt(n_cell, n_site)
                    print('data done')
                    # save(dirname=dirname, name=f'cell{n_cell}_site{n_site}_{i}', data=data)
                    reads, tg, tree = data
                    reads = add_noise(reads)

                    # t_start_cnj = time()
                    # copy_number_tree = run_copy_num_nj(reads)
                    # t_end_cnj = time()
                    # print('nj done')
                    # t_start_dice = time()
                    # dice_tree = run_dice(reads)
                    # t_end_dice = time()

                    t_start_scistree2 = time()
                    scistree2_tree, scistree2_geno = run_scistree2(reads)
                    t_end_scistree2 = time()
                    print('scistree2 done')

                    t_start_cellphy = time()
                    cellphy_tree = run_cellphy_reads(reads)
                    t_end_cellphy = time()
                    cellphy_geno = get_cellphy_genotype('cellphy_tmp', tg)
                    print('cellphy done')
                    # t_start_scistreec = time()
                    # scistreec_tree, scistreec_geno = run_scistreec(reads, scistree2_tree)
                    # t_end_scistreec = time()

                    # copy_number_tree_time = t_end_cnj - t_start_cnj
                    # dice_tree_time = t_end_dice - t_start_dice
                    scistree2_tree_time = t_end_scistree2 - t_start_scistree2
                    cellphy_tree_time = t_end_cellphy - t_start_cellphy
                    # scistreec_tree_time = t_end_scistreec - t_start_scistreec

                    results.extend([
                        # [n_cell, n_site, i, 'copy_nj', 'time', copy_number_tree_time],
                                    # [n_cell, n_site, i, 'dice', 'time', dice_tree_time],
                                    [n_cell, n_site, i, 'scistree2', 'time', scistree2_tree_time], 
                                    [n_cell, n_site, i, 'cellphy', 'time', cellphy_tree_time],
                                    # [n_cell, n_site, i, 'scistreec', 'time', scistreec_tree_time]
                                    ])

                except Exception as e:
                    pass
                print(f'complete {n_cell} {n_site}_{i}')
        #         break
        #     break
        # break
    df = pd.DataFrame(results, columns=['ncell', 'nsite', 'index', 'method', 'metric', 'value'])
    df.to_csv('scsim_runtime_500.csv', index=None)



if __name__ == "__main__":
    # run_simulation_cellcoal()
    # run_simulation_scsim_clt()
    # run_simulation_noise()
    # run_simulation_scsim_clone()
    run_simulation_runtime()
    # run_simulation_missing()
    # run_simulation_scsim_clt_low_coverage()
