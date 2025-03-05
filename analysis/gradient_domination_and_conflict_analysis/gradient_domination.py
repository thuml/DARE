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

attention_list = []
for i in tqdm(range(100)):
    attention_norm = np.sqrt(np.sum(attention_category_grad_two_embedding[i]*attention_category_grad_two_embedding[i], axis=1))
    attention_norm_mean = np.nanmean(np.ones_like(attention_norm)/(np.ones_like(attention_norm)/attention_norm))
    attention_list.append(attention_norm_mean)

representation_list = []
for i in tqdm(range(100)):
    representation_norm = np.sqrt(np.sum(representation_category_grad_two_embedding[i]*representation_category_grad_two_embedding[i], axis=1))
    representation_norm_mean = np.nanmean(np.ones_like(representation_norm)/(np.ones_like(representation_norm)/representation_norm))
    representation_list.append(representation_norm_mean)

print("attention norm:", sum(attention_list)/len(attention_list))
print("representation norm:", sum(representation_list)/len(representation_list))
