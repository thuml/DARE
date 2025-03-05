import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import re

plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams.update({'font.size': 18})
plt.rcParams['ytick.labelsize'] = 26

npy_dir = "gsu_result/taobao/simulate_gsu_case_study"
case_study_files = os.listdir(npy_dir)
if not os.path.exists("case_study"):
    os.mkdir("case_study")

if not os.path.exists("case_study/taobao"):
    os.mkdir("case_study/taobao")

for file in case_study_files[0:60]:
    nums = [int(num) for num in re.findall(r'\d+', file)]
    assert len(nums) == 2
    index = nums[0]
    aim_cate = nums[1]

    data_npy = np.load(os.path.join(npy_dir, file))

    data_npy[0] = data_npy[0][::-1]

    data_npy = data_npy[:, :10]

    data_npy[1:] = np.exp(data_npy[1:])

    data_npy = data_npy[np.array([0, 6, 1, 2, 3, 4, 5])]
    data_npy[1:] = (data_npy[1:] - np.min(data_npy[1:], axis=1)[:, np.newaxis]) / (np.max(data_npy[1:], axis=1) - np.min(data_npy[1:], axis=1))[:, np.newaxis]

    models = ['Ground Truth', 'DARE', 'TWIN-4E', 'TWIN', 'TWIN w/ Proj.', 'DIN']

    fig, ax = plt.subplots(figsize=[8.26667, 5.6])
    sns.heatmap(
        pd.DataFrame(np.round(data_npy[1:], 2), columns=data_npy[0][:10].astype(int), index=models),
        annot=True, xticklabels=True, yticklabels=True, square=True, cmap="Blues",
        cbar=False)
    ax.set_title('Correlations with Category {}'.format(aim_cate), fontsize=30)
    ax.set_xlabel('Behavior Sequence', fontsize=30)
    plt.tight_layout()
    plt.savefig(os.path.join("case_study/taobao", 'index_{}_category_{}.png'.format(index, aim_cate)), dpi=200, bbox_inches='tight')
    plt.close()


