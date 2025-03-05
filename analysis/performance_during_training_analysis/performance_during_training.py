import matplotlib.pyplot as plt
import argparse
import os
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams.update({'font.size': 22})

parser = argparse.ArgumentParser()
parser.add_argument('--twin_record_path', type=str, default='None')
parser.add_argument('--dare_record_path', type=str, default='None')
parser.add_argument('--output_path', type=str, default='None')
args = parser.parse_args()

with open(args.twin_record_path, 'r') as file:
    line_index = 0
    for line in file:
        line_index += 1
        if line_index>10:
            break
        if line_index == 8:
            number_str = line.strip('\n').strip('[').strip(']')
            numbers = number_str.split(',')
            x_numbers = [int(n.strip()) for n in numbers]
        elif line_index == 10:
            number_str = line.strip('\n').strip('[').strip(']')
            numbers = number_str.split(',')
            y_numbers1 = [float(n.strip()) for n in numbers]

with open(args.dare_record_path, 'r') as file:
    line_index = 0
    for line in file:
        line_index += 1
        if line_index>10:
            break
        if line_index == 10:
            number_str = line.strip('\n').strip('[').strip(']')
            numbers = number_str.split(',')
            y_numbers2 = [float(n.strip()) for n in numbers]


if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

# only draw one epoch
epoch_iteration = len(x_numbers)//2
plt.figure(figsize=(6.1, 5.6))
plt.plot(x_numbers[:epoch_iteration], y_numbers1[:epoch_iteration], label='TWIN', color='#5BAE7E')
plt.plot(x_numbers[:epoch_iteration], y_numbers2[:epoch_iteration], label='DARE', color='#F2A53A')
plt.legend(fontsize=20, framealpha=0.8, edgecolor='black', fancybox=False)
plt.xlabel("Training Iteration", fontsize=25)
plt.ylabel('Validation Accuracy', fontsize=25)
if "tmall" in args.output_path:
    plt.title("Training Performance on Tmall", fontsize=25, y=1.03)
    plt.yticks([0.84, 0.86, 0.88, 0.90, 0.92])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, 'result_tmall.pdf'))
    plt.savefig(os.path.join(args.output_path, 'result_tmall.png'))
else:
    plt.title("Training Performance on Taobao", fontsize=25, y=1.03)
    plt.yticks([0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88])
    plt.xticks([500, 1000, 1500, 2000])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_path, 'result_taobao.pdf'))
    plt.savefig(os.path.join(args.output_path, 'result_taobao.png'))

