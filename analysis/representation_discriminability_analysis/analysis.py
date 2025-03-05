import argparse
import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm


def calc_mutual_information():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("-seed", type=int, nargs='+', default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    representation = np.load(os.path.join(args.log_dir, 'repr', 'repr.npy'))

    print(representation.shape)

    target = np.load(os.path.join(args.log_dir, 'repr', 'target.npy'))

    for seed in args.seed:
        print("seed:", seed)
        n_clusters_list = [2, 5, 10, 20, 30, 50, 100, 200, 300, 500, 700, 1000]
        mi_list = []
        nmi_list = []
        for n_clusters in tqdm(n_clusters_list):
            km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed).fit(representation)
            discretization = np.array(km.labels_)
            mi = mutual_info_score(target, discretization)
            mi_list.append(mi)
            nmi = normalized_mutual_info_score(target, discretization)
            nmi_list.append(nmi)

        os.makedirs(os.path.join(args.log_dir, 'mutual_information_seed'+str(seed)), exist_ok=True)
        np.save(
            os.path.join(args.log_dir, 'mutual_information_seed'+str(seed), 'n_clusters_list.npy'),
            np.array(n_clusters_list)
        )
        np.save(
            os.path.join(args.log_dir, 'mutual_information_seed'+str(seed), 'mi_list.npy'),
            np.array(mi_list)
        )
        np.save(
            os.path.join(args.log_dir, 'mutual_information_seed'+str(seed), 'nmi_list.npy'),
            np.array(nmi_list)
        )


if __name__ == '__main__':
    calc_mutual_information()
