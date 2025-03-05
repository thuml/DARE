def sanity_check():
    per_user_ts_list = dict()
    dataset_path = '../data/tmall/user_log_sorted.csv'

    cnt = 0
    error_cnt = 0

    with open(dataset_path, 'r') as file:
        for line in file:
            uid, iid, cid, _, _, ts, btype = line.split(',')
            if btype == 'action_type\n' or int(btype) != 0:
                continue

            cnt += 1
            if uid not in per_user_ts_list:
                per_user_ts_list[uid] = []
            if len(per_user_ts_list[uid]) > 0 and int(ts) < per_user_ts_list[uid][-1]:
                error_cnt += 1
            per_user_ts_list[uid].append(int(ts))

            if cnt >= 200000:
                break

    print(error_cnt, cnt)


def sort_log():
    per_user_log = dict()
    dataset_path = '../data/tmall/user_log_format1.csv'

    with open(dataset_path, 'r') as file:
        for line in file:
            uid, iid, cid, _, _, ts, btype = line.split(',')
            if btype == 'action_type\n' or int(btype) != 0 or len(ts) != 4:
                continue
            if uid not in per_user_log:
                per_user_log[uid] = []
            per_user_log[uid].append([line, int(ts)])

    for uid in per_user_log:
        per_user_log[uid].sort(key=lambda x: x[1])
    sorted_log = []
    for uid in per_user_log:
        sorted_log.extend([x[0] for x in per_user_log[uid]])

    save_path = '../data/tmall/user_log_sorted.csv'
    with open(save_path, 'w') as file:
        for line in sorted_log:
            file.write(line)


if __name__ == '__main__':
    sort_log()
    sanity_check()
