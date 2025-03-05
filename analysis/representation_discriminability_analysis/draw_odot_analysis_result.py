import os
import sys
import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np

plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams.update({'font.size': 14})
sys.path.append('../../../..')
sys.path.append('../../..')

parser = argparse.ArgumentParser()
parser.add_argument('--force', default=False, action='store_true')
parser.add_argument('--dataset_name', default=None, type=str)
parser.add_argument('--twin_without_odot_path', default=None, type=str)
parser.add_argument('--twin_only_odot_path', default=None, type=str)
parser.add_argument('--two_embedding_without_odot_path', default=None, type=str)
parser.add_argument('--two_embedding_only_odot_path', default=None, type=str)
args = parser.parse_args()

from rliable import library as rly
from rliable import metrics

os.makedirs('buffer', exist_ok=True)
os.makedirs('visualization', exist_ok=True)

colors = [
    # '#444444',
    '#5BAE7E',
    # '#216DCE',
    '#4887D9',
    '#F2A53A',
    '#E67889',
]


def interval_estimates(score_dict, reps=50000):
    # scores: dict(str, (`num_runs` x `num_tasks` x ..))
    IQM = lambda scores: np.array([metrics.aggregate_mean(scores[..., frame])
                                   for frame in range(scores.shape[-1])])

    aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
        score_dict, IQM, reps=reps, random_state=np.random.RandomState(1234))

    return aggregate_scores, aggregate_interval_estimates

twin_without_odot_files = [os.path.join(args.twin_without_odot_path, "mutual_information_seed{}".format(i), 'nmi_list.npy') for i in range(1,6)]
twin_only_odot_files = [os.path.join(args.twin_only_odot_path, "mutual_information_seed{}".format(i), 'nmi_list.npy') for i in range(1,6)]
DARE_without_odot_files = [os.path.join(args.two_embedding_without_odot_path, "mutual_information_seed{}".format(i), 'nmi_list.npy') for i in range(1,6)]
DARE_only_odot_files = [os.path.join(args.two_embedding_only_odot_path, "mutual_information_seed{}".format(i), 'nmi_list.npy') for i in range(1,6)]

method_name_list = ["twin_without_odot", "twin_only_odot", "two_embedding_without_odot", "two_embedding_only_odot"]
plt_label_list = ["TWIN $[r, v_t]$", "TWIN $[r \odot v_t]$", "DARE $[r, v_t]$", "DARE $[r \odot v_t]$"]

plt.figure(figsize=(6.2, 5.4))
for idx, aim_files in enumerate([twin_without_odot_files, twin_only_odot_files, DARE_without_odot_files, DARE_only_odot_files]):
    curves = []
    for file_idx, one_file in enumerate(aim_files):
        steps = np.load(os.path.join(args.twin_without_odot_path, "mutual_information_seed{}".format(file_idx+1), 'n_clusters_list.npy'))
        steps = np.log2(steps)
        values = np.load(one_file)
        curves.append(values)

    minlen = min([len(v) for v in curves])
    curves = [v[:minlen] for v in curves]
    curves = np.stack(curves)
    print("curves shape:", curves.shape)

    method_name = method_name_list[idx]
    plt_label = plt_label_list[idx]
    max_cluser_num = 4 if args.dataset_name=='Taobao' else 6
    pkl_path = f'buffer/{args.dataset_name}_{method_name}.pkl'

    if not args.force and os.path.exists(pkl_path):
        _, aggregate_scores, aggregate_interval_estimates = pickle.load(open(pkl_path, 'rb'))
        topass = (minlen == len(aggregate_scores))
    else:
        topass = False


    if args.force or (not topass and (curves.shape[0] >= 4 or True)):
        scores_dict = {method_name: curves[:, None, :]}  # (8, 1, 26)
        aggregate_scores, aggregate_interval_estimates = interval_estimates(scores_dict)
        aggregate_scores = aggregate_scores[method_name]
        aggregate_interval_estimates = aggregate_interval_estimates[method_name]

        pickle.dump((steps[:len(aggregate_scores)], aggregate_scores, aggregate_interval_estimates),
                    open(pkl_path, 'wb'))

    steps, aggregate_scores, aggregate_interval_estimates = pickle.load(open(pkl_path, 'rb'))
    steps = steps[:len(aggregate_scores)]

    plt.plot(
        steps[:-max_cluser_num], aggregate_scores[:-max_cluser_num], "-", linewidth=2, label=plt_label, color=colors[idx]
    )
    print(aggregate_scores)
    print(steps)
    plt.fill_between(
        steps[:-max_cluser_num],
        aggregate_interval_estimates[0][:-max_cluser_num],
        aggregate_interval_estimates[1][:-max_cluser_num],
        alpha=0.1,
        color=colors[idx]
    )


if args.dataset_name=='Taobao' or args.dataset_name=='taobao':
    plt.xlim(1, 7.65)
    plt.xticks([1, 2, 3, 4, 5, 6, 7], ['$2^1$', '$2^2$', '$2^3$', '$2^4$', '$2^5$', '$2^6$', '$2^7$'])
else:
    plt.xlim(1, 5.65)
    plt.xticks([1, 2, 3, 4, 5], ['$2^1$', '$2^2$', '$2^3$', '$2^4$', '$2^5$'])

plt.legend(fontsize=16, framealpha=0.8, edgecolor='black', fancybox=False)
plt.ylabel('Mutual Information', fontsize=18)
plt.xlabel('Number of Clusters', fontsize=18)
plt.title(f"Representation's Discriminability on {args.dataset_name}", fontsize=20, y=1.03)
plt.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'visualization/{args.dataset_name}_odot_analysis.png')
plt.clf()
