import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
plt.rcParams["font.family"] = "Arial"
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams.update({'font.size': 22})

attention_category_grad_two_embedding = np.load("../../log/taobao/model_twin_stor_grad/grad_during_training/attn_category_grad.npy")
representation_category_grad_two_embedding = np.load("../../log/taobao/model_twin_stor_grad/grad_during_training/repr_category_grad.npy")
print(attention_category_grad_two_embedding.shape)
angle_list = []
for i in tqdm(range(attention_category_grad_two_embedding.shape[0])):
    category_angle = np.sum(attention_category_grad_two_embedding[i]*representation_category_grad_two_embedding[i], axis=1)\
            /np.sqrt(np.sum(attention_category_grad_two_embedding[i]**2, axis=1))/np.sqrt(np.sum(representation_category_grad_two_embedding[i]**2, axis=1))
    category_angle = np.nanmean(np.ones_like(category_angle)/(np.ones_like(category_angle)/category_angle))
    angle_list.append(category_angle)
weights=np.full(len(angle_list),fill_value=1/len(angle_list))

plt.figure(figsize=(6.0, 4.0))
plt.hist(angle_list, bins=40, weights=weights, edgecolor='black', range=(-1,1))
plt.xticks(ticks=[-1, -0.5, 0, 0.5, 1])
plt.xlim(-1, 1)
#plt.yticks(ticks=[0, 1, 2, 3])
# use percentage as ticks e.g. 15%
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1,decimals=0))
plt.grid(alpha=0.2)
plt.axvline(x=0, color='r')
plt.axvspan(-1, 0, color='red', alpha=0.1)
plt.axvspan(0, 1, color='green', alpha=0.1)
plt.ylabel('Proportion', fontsize=24)
plt.xlabel('Cosine Similarity', fontsize=24)
#plt.title("Gradient Relation", fontsize=20, y=1.03)
plt.text(-0.9, 0.1, 'Conflicting', fontsize=21, color='red')
plt.text(0.21, 0.1, 'Consistent', fontsize=21, color='green')

conflict_percentage = len([i for i in angle_list if i < 0])/len(angle_list)
consistent_percentage = len([i for i in angle_list if i >= 0])/len(angle_list)

plt.text(-0.82, 0.07, f'{conflict_percentage*100:.2f}%', fontsize=21, color='red')
plt.text(0.29, 0.07, f'{consistent_percentage*100:.2f}%', fontsize=21, color='green')
plt.tight_layout()

if not os.path.exists("./gradient_analysis_visualization"):
    os.mkdir("./gradient_analysis_visualization")
plt.savefig("./gradient_analysis_visualization/angle_distribution.pdf")
plt.savefig("./gradient_analysis_visualization/angle_distribution.png")

