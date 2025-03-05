import argparse
import os

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def gauss_normalize(x):
    return (x - np.mean(x)) / np.std(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='tin')
    parser.add_argument('--c_embedding_path', type=str, default=None)
    parser.add_argument('--p_embedding_path', type=str, default=None)
    parser.add_argument('--c_embedding_path_target', type=str, default=None)
    parser.add_argument('--p_embedding_path_target', type=str, default=None)
    parser.add_argument('--most_common_cid_path', type=str)
    parser.add_argument('--vis_nc', type=int, default=10)
    parser.add_argument('--vis_np', type=int, default=10)
    parser.add_argument('--figsize', type=int, nargs='+', default=[5, 5])
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    model = args.model
    c_embedding_path = args.c_embedding_path
    p_embedding_path = args.p_embedding_path
    c_embedding_path_target = args.c_embedding_path_target
    p_embedding_path_target = args.p_embedding_path_target
    vis_nc = args.vis_nc
    vis_np = args.vis_np
    figsize = args.figsize
    save_dir = args.save_dir
    most_common_cid = np.load(args.most_common_cid_path)
    os.makedirs(save_dir, exist_ok=True)

    for idx, target in enumerate(most_common_cid):
        attn = np.zeros((vis_nc, vis_np))
        repre = np.zeros((vis_nc, vis_np))
        mutual_information = np.zeros((vis_nc, vis_np))

        if model == 'gsu':
            c_embedding = np.load(c_embedding_path)
            p_embedding = np.load(p_embedding_path)
            if c_embedding_path_target is not None:
                c_embedding_target = np.load(c_embedding_path_target)
            else:
                c_embedding_target = c_embedding
            if p_embedding_path_target is not None:
                p_embedding_target = np.load(p_embedding_path_target)
            else:
                p_embedding_target = p_embedding
            for i in range(vis_nc):
                cid = most_common_cid[i]
                for p in range(1, vis_np + 1):
                    category_score = (c_embedding_target[target] * c_embedding[cid]) / (len(c_embedding_target[target]) ** 0.5)
                    time_score = (p_embedding_target[0] * p_embedding[p]) / (len(p_embedding_target[0]) ** 0.5)
                    attn[i][p-1] = np.sum(category_score) + np.sum(time_score)
                    repre[i][p - 1] = 1
        else:
            assert False, 'model type error'

        attn_logit = attn
        attn = softmax(attn)
        attn = min_max_normalize(attn)

        np.save(os.path.join(save_dir, 'model_{}_idx_{}_category_{}_norm_attn.npy'.format(model, idx, target)), attn)

        ind = [str(p + 1) for p in range(vis_np)]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pd.DataFrame(np.round(attn, 2), columns=ind, index=most_common_cid[:vis_nc]),
            annot=True, xticklabels=True, yticklabels=True, square=True, cmap="Blues",
            cbar=False)
        ax.set_title('Learned attention given target category ID {}'.format(target), fontsize=12)
        ax.set_ylabel('Top-{} appeared history behaviors'.format(vis_nc))
        ax.set_xlabel('Target-relative Position')
        plt.savefig(os.path.join(save_dir, 'model_{}_idx_{}_category_{}_attn.png'.format(model, idx, target)),
                    dpi=200,
                    bbox_inches='tight')
        plt.close()
