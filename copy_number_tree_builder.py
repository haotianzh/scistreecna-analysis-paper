from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from Bio import Phylo
import numpy as np
from io import StringIO
from sklearn.cluster import KMeans


def construct_nj_tree_with_biopython(copy_numbers):
    n_cells = copy_numbers.shape[1]

    pairwise_distances = []
    for i in range(n_cells):
        row_distances = []
        for j in range(i):  # Only compute distances for the lower triangle (j < i)
            distance = np.linalg.norm(copy_numbers[:, i] - copy_numbers[:, j])
            row_distances.append(distance)
        pairwise_distances.append(row_distances)

 
    names = [f"{i}" for i in range(n_cells)]
    distance_matrix = DistanceMatrix(names)

    # Fill the lower triangle of the DistanceMatrix
    for i, row_distances in enumerate(pairwise_distances):
        for j, distance in enumerate(row_distances):
            distance_matrix[i, j] = distance


    constructor = DistanceTreeConstructor()
    nj_tree = constructor.nj(distance_matrix)
    for clade in nj_tree.get_nonterminals():
        clade.name = None
    newick_string = nj_tree.format('newick')
    import re
    newick = re.sub(r':[^,);]+', '', newick_string)
    return newick



def cluster_and_average_copy_numbers(copy_numbers, k=8):
    copy_number_vectors = copy_numbers.T  # Shape: (n_cells, n_sites)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(copy_number_vectors)

    averaged_copy_numbers = np.zeros_like(copy_numbers.T)  # Shape: (n_cells, n_sites)
    for cluster in range(k):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_average = np.mean(copy_number_vectors[cluster_indices], axis=0)
        averaged_copy_numbers[cluster_indices] = cluster_average

    return averaged_copy_numbers.T

if __name__ == "__main__":
    cnum = np.random.randint(low=1, high=4, size=(100, 50))
    # tree = construct_nj_tree_with_biopython(cnum)
    # print(tree)
    
    # Example usage
    k = 8
    averaged_copy_numbers = cluster_and_average_copy_numbers(cnum, k=k)
    print("Averaged Copy Numbers:")
    print(averaged_copy_numbers)