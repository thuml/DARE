import os
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import AnalysisDataset


def compute_co_occurrence(train_dataset_path, test_dataset_path, test_dataset_only, n_category, save_dir):
    print('Computing co-occurrence')
    if os.path.exists(os.path.join(save_dir, 'adj_p.npy')) and \
            os.path.exists(os.path.join(save_dir, 'adj_n.npy')) and \
            os.path.exists(os.path.join(save_dir, 'target_p.npy')) and \
            os.path.exists(os.path.join(save_dir, 'target_n.npy')):
        return

    adj_p = {}
    adj_n = {}
    target_p = np.zeros(n_category)
    target_n = np.zeros(n_category)
    if test_dataset_only:
        all_dataset = np.load(test_dataset_path, allow_pickle=True)
    else:
        all_dataset = AnalysisDataset(train_dataset_path, test_dataset_path)

    for one_data in tqdm.tqdm(all_dataset):
        _, iid, cid, label, hist_iid, hist_cid = one_data[:6]
        hist_len = len(hist_iid)
        if label[0] == 1:
            for j in range(len(hist_iid)):
                if hist_iid[j] == 0:
                    continue
                if (cid, hist_cid[j], hist_len - j) in adj_p:
                    adj_p[(cid, hist_cid[j], hist_len - j)] += 1
                else:
                    adj_p[(cid, hist_cid[j], hist_len - j)] = 1
            target_p[cid] += 1
        elif label[0] == 0:
            for j in range(len(hist_iid)):
                if hist_iid[j] == 0:
                    continue
                if (cid, hist_cid[j], hist_len - j) in adj_n:
                    adj_n[(cid, hist_cid[j], hist_len - j)] += 1
                else:
                    adj_n[(cid, hist_cid[j], hist_len - j)] = 1
            target_n[cid] += 1

    adj_p = np.asarray(adj_p, dtype=object)
    np.save(os.path.join(save_dir, 'adj_p.npy'), adj_p)
    adj_n = np.asarray(adj_n, dtype=object)
    np.save(os.path.join(save_dir, 'adj_n.npy'), adj_n)
    target_p = np.asarray(target_p, dtype=object)
    np.save(os.path.join(save_dir, 'target_p.npy'), target_p)
    target_n = np.asarray(target_n, dtype=object)
    np.save(os.path.join(save_dir, 'target_n.npy'), target_n)


def compute_most_common(train_dataset_path, test_dataset_path, test_dataset_only, n_most_common, save_dir):
    print('Computing most common category')
    if os.path.exists(os.path.join(save_dir, 'most_common_{}_category.npy'.format(n_most_common))):
        return

    if test_dataset_only:
        all_dataset = np.load(test_dataset_path, allow_pickle=True)
    else:
        all_dataset = AnalysisDataset(train_dataset_path, test_dataset_path)

    cid_list = []
    for one_data in tqdm.tqdm(all_dataset):
        _, _, cid, _, _, _, = one_data[:6]
        cid_list.append(cid)
    cid_counter = Counter(cid_list)
    print(cid_counter.most_common(n_most_common))
    most_common_cid = [cid for cid, _ in cid_counter.most_common(n_most_common)]
    np.save(os.path.join(save_dir, 'most_common_{}_category.npy'.format(n_most_common)), most_common_cid)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def plot_mutual_information(n_most_common, save_dir, vis_nc, vis_np, figsize, normalize_mi_mat):
    # hack
    # distribution c x c x l
    # all sample 1 x l x N
    # only reasonable when N >> c x c
    print('Plotting mutual information')
    adj_p = np.load(os.path.join(save_dir, 'adj_p.npy'), allow_pickle=True).item()
    adj_n = np.load(os.path.join(save_dir, 'adj_n.npy'), allow_pickle=True).item()
    target_p = np.load(os.path.join(save_dir, 'target_p.npy'), allow_pickle=True)
    target_n = np.load(os.path.join(save_dir, 'target_n.npy'), allow_pickle=True)
    most_common_cid = np.load(os.path.join(save_dir, 'most_common_{}_category.npy'.format(n_most_common)))

    save_root = os.path.join(save_dir, 'gt_mi')
    os.makedirs(save_root, exist_ok=True)

    corr_mat = np.zeros((vis_nc, vis_nc))
    for idx, target in enumerate(most_common_cid):
        mat = np.zeros((vis_nc, vis_np))
        t_p = target_p[target]
        t_n = target_n[target]
        s = t_p + t_n

        for i in range(vis_nc):
            for p in range(1, vis_np + 1):
                if (target, most_common_cid[i], p) in adj_p:
                    x_p = adj_p[(target, most_common_cid[i], p)]
                else:
                    x_p = 1e-5
                if (target, most_common_cid[i], p) in adj_n:
                    x_n = adj_n[(target, most_common_cid[i], p)]
                else:
                    x_n = 1e-5

                m1 = x_p / s * np.log2(x_p * s / ((x_p + x_n) * t_p))  # x_i = 1, y = 1
                m2 = x_n / s * np.log2(x_n * s / ((x_p + x_n) * t_n))  # x_i = 1, y = 0
                m3 = (t_p - x_p) / s * np.log2((t_p - x_p) * s / ((t_p + t_n - x_p - x_n) * t_p))  # x_i = 0, y = 1
                m4 = (t_n - x_n) / s * np.log2((t_n - x_n) * s / ((t_p + t_n - x_p - x_n) * t_n))  # x_i = 0, y = 0

                m = m1 + m2 + m3 + m4
                mat[i][p - 1] = m

            corr_mat[idx][i] = mat[i][0]

        if normalize_mi_mat:
            mat = normalization(mat)

        np.save(os.path.join(save_root, 'idx_{}_category_{}_norm.npy'.format(idx, target)), mat)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pd.DataFrame(np.round(mat, 2), columns=[i + 1 for i in range(vis_np)], index=most_common_cid[:vis_nc]),
            annot=True, xticklabels=True, yticklabels=True, square=True, cmap="Blues",
            cbar=False)

        ax.set_title('Category-wise Target-aware Correlation', fontsize=12)
        ax.set_xlabel('Target-relative Position', fontsize=10)
        ax.set_ylabel('Top-{} behavior categories'.format(vis_nc), fontsize=10)
        plt.savefig(os.path.join(save_root, 'idx_{}_category_{}.png'.format(idx, target)), dpi=200, bbox_inches='tight')
        plt.close()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pd.DataFrame(np.round(corr_mat, 2), columns=most_common_cid[:vis_nc], index=most_common_cid[:vis_nc]),
        annot=True, xticklabels=True, yticklabels=True, square=True, cmap="Blues",
        cbar=False)
    ax.set_title('Category-wise Correlation', fontsize=12)
    plt.savefig(os.path.join(save_root, 'category_correlation.png'), dpi=200, bbox_inches='tight')
    plt.close()

    corr_mat = corr_mat / np.max(corr_mat, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pd.DataFrame(np.round(corr_mat, 2), columns=most_common_cid[:vis_nc], index=most_common_cid[:vis_nc]),
        annot=True, xticklabels=True, yticklabels=True, square=True, cmap="Blues",
        cbar=False)
    ax.set_title('Normalized Category-wise Correlation', fontsize=12)
    plt.savefig(os.path.join(save_root, 'category_correlation_normalized.png'), dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', type=str, default='../data/book_data/remap_book_train.npy')
    parser.add_argument('--test_dataset_path', type=str, default='../data/book_data/remap_book_test.npy')
    parser.add_argument('--test_dataset_only', action='store_true')
    parser.add_argument('--n_category', type=int, default=1576)
    parser.add_argument('--n_most_common', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='attention_accuracy_result')
    parser.add_argument('--vis_nc', type=int, default=10)
    parser.add_argument('--vis_np', type=int, default=10)
    parser.add_argument('--figsize', type=int, nargs='+', default=[5, 5])
    parser.add_argument('--normalize_mi_mat', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    compute_co_occurrence(
        args.train_dataset_path,
        args.test_dataset_path,
        args.test_dataset_only,
        args.n_category,
        args.save_dir
    )
    compute_most_common(
        args.train_dataset_path,
        args.test_dataset_path,
        args.test_dataset_only,
        args.n_most_common,
        args.save_dir
    )
    plot_mutual_information(
        args.n_most_common,
        args.save_dir,
        args.vis_nc,
        args.vis_np,
        args.figsize,
        args.normalize_mi_mat
    )
