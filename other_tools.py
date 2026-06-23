import os 
import popgen
import numpy as np 

def write_to_cellphy(matrix):
    nsite, ncell = matrix.shape
    output = f'cellphy_tmp.vcf'
    header = \
f'''##fileformat=VCFv4.3
##fileDate=NOW
##source=ov2295
##ncell={ncell}
##nsite={nsite}
##reference=NONE
##contig=<ID=1>
##phasing=NO
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phread-scaled genotype likelihoods">
'''
    with open(output, 'w') as out:
        out.write(header)
        out.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT')
        for i in range(ncell):
            out.write(f'\t{i}')
        out.write('\n')
        for idx, i in enumerate(range(nsite)):
            chrom, coord, ref, alt, af, snp_id = 1, idx, 'A', 'T', 0.1, idx
            out.write(f"{chrom}\t{coord}\t{snp_id}\t{ref}\t{alt}\t.\tPASS\tAF={af}\tGT:PL")
            for j in range(ncell):
                out.write(f'\t{matrix[i, j]}')
            out.write('\n')    



# def phred_likelihood_with_ado_seqerr_gt(ref_counts, alt_counts, ado=0.2, seqerr=0.01):
#     # Q-phred score = -10 * log10(p)
#     p00, p01, p10, p11 = np.log(1-seqerr), np.log(seqerr), np.log(seqerr), np.log(1-seqerr)
#     l00 = np.exp((ref_counts*p00 + alt_counts*p01).astype(float))
#     l01 = (1-ado)*np.exp((ref_counts*np.log(0.5*np.exp(p00)+0.5*np.exp(p10))).astype(float)+\
#                          (alt_counts*np.log(0.5*np.exp(p01)+0.5*np.exp(p11))).astype(float))+\
#             (0.5*ado)*(np.exp((ref_counts*p00+alt_counts*p10).astype(float))+\
#                        np.exp((ref_counts*p10+alt_counts*p11).astype(float)))
#     l11 = np.exp((ref_counts*p10 + alt_counts*p11).astype(float))
#     q00 = -10 * np.log10(l00)
#     q11 = -10 * np.log10(l11)
#     q01 = -10 * np.log10(l01)
#     q00 = q00.astype(int)
#     q11 = q11.astype(int)
#     q01 = q01.astype(int)
#     return q00, q01, q11


def phred_likelihood_with_ado_seqerr_gt(ref_counts, alt_counts, ado=0.2, seqerr=0.01):
    # Q-phred score = -10 * log10(p)
    p00, p01, p10, p11 = np.log(1-seqerr), np.log(seqerr), np.log(seqerr), np.log(1-seqerr)
    l00 = np.exp((ref_counts*p00 + alt_counts*p01).astype(np.float128))
    l01 = (1-ado)*np.exp((ref_counts*np.log(0.5*np.exp(p00)+0.5*np.exp(p10))).astype(np.float128)+\
                         (alt_counts*np.log(0.5*np.exp(p01)+0.5*np.exp(p11))).astype(np.float128))+\
            (0.5*ado)*(np.exp((ref_counts*p00+alt_counts*p10).astype(np.float128))+\
                       np.exp((ref_counts*p10+alt_counts*p11).astype(np.float128)))
    l11 = np.exp((ref_counts*p10 + alt_counts*p11).astype(np.float128))
    q00 = -10 * np.log10(l00)
    q11 = -10 * np.log10(l11)
    q01 = -10 * np.log10(l01)
    q00 = q00.astype(int)
    q11 = q11.astype(int)
    q01 = q01.astype(int)
    return q00, q01, q11




def get_ml_gt(ref_counts, alt_counts, ado=0.2):
    # a, b, c = phred_likelihood_with_fn_fp_flat(ref_counts, alt_counts)
    a, b, c = phred_likelihood_with_ado_seqerr_gt(ref_counts, alt_counts, ado=ado)
    d = np.concatenate([a[:, :, np.newaxis], b[:, :, np.newaxis], c[:, :, np.newaxis]], axis=-1)
    arg_max = np.argmin(d, axis=-1)
    return arg_max
    


def get_phred_likelihood(a, b, c, ml_gt):
    gts = ['0/0', '0/1', '1/1']
    n, m = a.shape
    mat = []
    for i in range(n):
        res = []
        for j in range(m):
            if a[i,j] == b[i,j] == c[i,j] == 0:
                res.append(f'./.:{a[i,j]},{b[i,j]},{c[i,j]}')
            else:
                res.append(f'{gts[ml_gt[i,j]]}:{a[i,j]},{b[i,j]},{c[i,j]}')
        mat.append(res)
    return np.array(mat)


def run_cellphy(dir, i):
    os.system(f'/home/haz19024/softwares/cellphy/cellphy.sh FAST -a -t 30 -r {dir}/{i}.vcf > {dir}/{i}.cellphy.log 2>&1')
    with open(f'{dir}/{i}.vcf.raxml.bestTree') as f:
        tree = f.readline().strip()
    tree = popgen.utils.from_newick(tree)
    tree = popgen.utils.relabel(tree, offset=-1)
    return tree


def run_cellphy_reads(reads):
    ref_cnts = reads[:, :, 0]
    alt_cnts = reads[:, :, 1]
    a, b, c = phred_likelihood_with_ado_seqerr_gt(ref_cnts, alt_cnts)
    
    gt = get_ml_gt(ref_cnts, alt_cnts)
    res = get_phred_likelihood(a, b, c, gt)
    write_to_cellphy(res)
    os.system(f'/home/haz19024/softwares/cellphy/cellphy.sh FAST -t 10 -r cellphy_tmp.vcf > cellphy_tmp.log 2>&1')
    # os.system(f'/home/haz19024/softwares/cellphy/cellphy.sh SEARCH -t 30 -r cellphy_tmp.vcf > cellphy_tmp.log 2>&1')
    # os.system(f'/home/haz19024/softwares/cellphy/cellphy.sh FAST -t 30 -r --prob-msa off cellphy_tmp.vcf > cellphy_tmp.log 2>&1')
    with open(f'cellphy_tmp.vcf.raxml.bestTree') as f:
        tree = f.readline().strip()
    tree = popgen.utils.from_newick(tree)
    # tree = popgen.utils.relabel(tree, offset=-1)
    return tree


def random_reads(n_leaves=10, n_sites=1):
    reads = []
    for site in range(n_sites):
        read = []
        for leave in range(n_leaves):
            ref = np.random.randint(low=0, high=5)
            alt = np.random.randint(low=0, high=5)
            read.append((ref, alt, 2))
            # read.append((0, 5, 2))
        reads.append(read)
    return np.array(reads)


if __name__ == "__main__":
    reads = random_reads(10, 5)
    ref_cnts = reads[:, :, 0]
    alt_cnts = reads[:, :, 1]

    ref_cnts = np.zeros([1, 10])
    alt_cnts = np.zeros([1, 10])
    a, b, c = phred_likelihood_with_ado_seqerr_gt(ref_cnts, alt_cnts, ado=0.5)
    gt = get_ml_gt(ref_cnts, alt_cnts, ado=0.5)
    print(a, b, c)
    # res = get_phred_likelihood(a, b, c, gt)
    # write_to_cellphy(res)