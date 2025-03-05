import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stixsans"
plt.rcParams.update({'font.size': 24})

colors = [
    '#5BAE7E',
    # '#216DCE',
    '#4887D9',
    '#F2A53A',
    '#444444',
    '#E67889',
]

x = [i for i in range(21)]
result = []
with open("./gsu_result/taobao/aim_result_ndcg.txt", "r") as file:
    meet_num = 0
    for line in file:
        if "ndcg" in line:
            line = line.strip("ndcg: ").split(" ")
            assert len(line) == 20
            line = [0]+[float(one_item) for one_item in line]
            result.append(line)

plt.figure(figsize=(6.2, 5.6))
plt.plot(x, result[0], label="DARE", color=colors[0])
plt.plot(x, result[1], label="TWIN-4E", color=colors[1])
plt.plot(x, result[2], label="TWIN", color=colors[2])
plt.plot(x, result[3], label="TWIN w/ Proj.", color=colors[3])
plt.plot(x, result[4], label="DIN", color=colors[4])
# plt.plot(x, result[5], label="Ground truth")
plt.xticks([0, 5, 10, 15, 20])
plt.yticks([0, 0.3, 0.6, 0.9])
plt.xlim(0, 20)
plt.ylim(0, 0.9)
plt.legend(fontsize=18, framealpha=0.8, edgecolor='black', fancybox=False)
plt.ylabel('NDCG', fontsize=26)
plt.xlabel('Behaviors Returned by Search', fontsize=26)
plt.title(f"Retrieval Performance on Taobao", fontsize=26, y=1.03)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./gsu_result/taobao/taobao_gsu_result.pdf")
plt.savefig("./gsu_result/taobao/taobao_gsu_result.png")
plt.cla()


x = [i for i in range(21)]
result = []
with open("./gsu_result/tmall/aim_result_ndcg.txt", "r") as file:
    meet_num = 0
    for line in file:
        if "ndcg" in line:
            line = line.strip("ndcg: ").split(" ")
            assert len(line) == 20
            line = [0]+[float(one_item) for one_item in line]
            result.append(line)

plt.figure(figsize=(6.2, 5.6))
plt.plot(x, result[0], label="DARE", color=colors[0])
plt.plot(x, result[1], label="TWIN-4E", color=colors[1])
plt.plot(x, result[2], label="TWIN", color=colors[2])
plt.plot(x, result[3], label="TWIN w/ Proj.", color=colors[3])
plt.plot(x, result[4], label="DIN", color=colors[4])
# plt.plot(x, result[5], label="Ground truth")
plt.xticks([0, 5, 10, 15, 20])
plt.yticks([0, 0.3, 0.6, 0.9])
plt.xlim(0, 20)
plt.ylim(0, 0.9)
plt.legend(fontsize=18, framealpha=0.8, edgecolor='black', fancybox=False)
plt.ylabel('NDCG', fontsize=26)
plt.xlabel('Behaviors Returned by Search', fontsize=26)
plt.title(f"Retrieval Performance on Tmall", fontsize=26, y=1.03)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./gsu_result/tmall/tmall_gsu_result.pdf")
plt.savefig("./gsu_result/tmall/tmall_gsu_result.png")
plt.cla()


