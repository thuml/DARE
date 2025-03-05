import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams.update({'font.size': 14})

# taobao
taobao_twin_grad_paths = [
    "../../log/taobao/model_twin_stor_grad/grad_during_training/attn_category_grad.npy",
    "../../log/taobao/model_twin_stor_grad/grad_during_training/attn_time_grad.npy",
    "../../log/taobao/model_twin_stor_grad/grad_during_training/repr_category_grad.npy",
    "../../log/taobao/model_twin_stor_grad/grad_during_training/repr_time_grad.npy",
]

name = 'taobao_twin_grad_paths'

attention_category_grad_path = taobao_twin_grad_paths[0]
attention_time_grad_path = taobao_twin_grad_paths[1]
representation_category_grad_path = taobao_twin_grad_paths[2]
representation_time_grad_path = taobao_twin_grad_paths[3]

attention_category_grad_np = np.load(attention_category_grad_path)
attention_time_grad_np = np.load(attention_time_grad_path)[:, :200, :]
representation_category_grad_np = np.load(representation_category_grad_path)
representation_time_grad_np = np.load(representation_time_grad_path)[:, :200, :]

conflict_ratio_record = []
record_num = attention_category_grad_np.shape[0]
dot_result_record = []
for i in range(100):
    iteration_attention_category_grad_np = attention_category_grad_np[i]
    iteration_representation_category_grad_np = representation_category_grad_np[i]
    iteration_attention_time_grad_np = attention_time_grad_np[i]
    iteration_representation_time_grad_np = representation_time_grad_np[i]

    dot_result = np.sum(iteration_attention_category_grad_np * iteration_representation_category_grad_np, axis=1)
    dot_result_record.append(dot_result)
    conflict_num = np.sum((dot_result < 0).astype(np.int32))
    not_conflict_num = np.sum((dot_result > 0).astype(np.int32))
    conflict_ratio_record.append(conflict_num / (conflict_num+not_conflict_num))

dot_result_record = np.stack(dot_result_record)

not_conflict_per_category = np.sum((dot_result_record>0), axis=0)
conflict_per_category = np.sum((dot_result_record<0), axis=0)
conflict_ratio_per_category = conflict_per_category / (conflict_per_category + not_conflict_per_category)
have_grad_number = conflict_per_category + not_conflict_per_category

category_num = len(have_grad_number)
category_conflict_ratio_occure_time_record = {}
for j in range(10):
    category_conflict_ratio_occure_time_record[j] = []
for i in range(category_num):
    conflict_ratio = conflict_ratio_per_category[i]
    if conflict_ratio > 0.9:
        category_conflict_ratio_occure_time_record[9].append(have_grad_number[i])
    elif 0.8 < conflict_ratio <= 0.9:
        category_conflict_ratio_occure_time_record[8].append(have_grad_number[i])
    elif 0.7 < conflict_ratio <= 0.8:
        category_conflict_ratio_occure_time_record[7].append(have_grad_number[i])
    elif 0.6 < conflict_ratio <= 0.7:
        category_conflict_ratio_occure_time_record[6].append(have_grad_number[i])
    elif 0.5 < conflict_ratio <= 0.6:
        category_conflict_ratio_occure_time_record[5].append(have_grad_number[i])
    elif 0.4 < conflict_ratio <= 0.5:
        category_conflict_ratio_occure_time_record[4].append(have_grad_number[i])
    elif 0.3 < conflict_ratio <= 0.4:
        category_conflict_ratio_occure_time_record[3].append(have_grad_number[i])
    elif 0.2 < conflict_ratio <= 0.3:
        category_conflict_ratio_occure_time_record[2].append(have_grad_number[i])
    elif 0.1 < conflict_ratio <= 0.2:
        category_conflict_ratio_occure_time_record[1].append(have_grad_number[i])
    elif conflict_ratio <= 0.1:
        category_conflict_ratio_occure_time_record[0].append(have_grad_number[i])
    else:
        assert np.isnan(conflict_ratio)

print(name)
for j in range(10):
    print(f"conflict ratio {j}0% to {j+1}0%, number: {len(category_conflict_ratio_occure_time_record[j])}, average_len: {sum(category_conflict_ratio_occure_time_record[j])/max(len(category_conflict_ratio_occure_time_record[j]), 1)}")

conflict_times = 0
for j in range(5, 10):
    conflict_times += len(category_conflict_ratio_occure_time_record[j])

total_times = 0
for j in range(0, 10):
    total_times += len(category_conflict_ratio_occure_time_record[j])

print("conflict_ratio:", conflict_times/total_times)

data_list = []
for draw_i in range(10):
    for _ in range(len(category_conflict_ratio_occure_time_record[draw_i])):
        data_list.append(draw_i*10+5)

weights=np.full(len(data_list),fill_value=1/len(data_list))
plt.hist(data_list, bins=10, weights=weights, edgecolor='black', range=(0,100))
plt.xticks(ticks=[0, 25, 50, 75, 100])
plt.axvline(x=50, color='r')

plt.xlabel('Conflict Ratio(%)', fontsize=22)
plt.ylabel('Category Ratio', fontsize=22)
plt.tight_layout()
plt.savefig(f"./gradient_analysis_visualization/taobao_grad_conflict_per_category.png")
plt.clf()

