from collections import Counter
import copy
import os
import random
import numpy as np
from tqdm import tqdm


def pad_and_truncate(seq, max_len=200):
    seq = copy.deepcopy(seq)
    if len(seq) < max_len:
        seq = [0] * (max_len - len(seq)) + seq
    if len(seq) > max_len:
        seq = seq[-max_len:]
    return seq


if __name__ == '__main__':
    dataset_path = '../data/taobao/UserBehavior.csv'

    if not os.path.exists('../data/taobao/uid_mapping.npy') or \
            not os.path.exists('../data/taobao/iid_mapping.npy') or \
            not os.path.exists('../data/taobao/cid_mapping.npy'):
        uid_mapping, iid_mapping, cid_mapping = {}, {}, {}

        with open(dataset_path, 'r') as file:
            for line in file:
                uid, iid, cid, btype, ts = line.split(',')
                if btype != 'pv':
                    continue

                uid, iid, cid = list(map(int, [uid, iid, cid]))
                if uid not in uid_mapping:
                    uid_mapping[uid] = len(uid_mapping) + 1
                if iid not in iid_mapping:
                    iid_mapping[iid] = len(iid_mapping) + 1
                if cid not in cid_mapping:
                    cid_mapping[cid] = len(cid_mapping) + 1

        np.save('../data/taobao/uid_mapping.npy', uid_mapping)
        np.save('../data/taobao/iid_mapping.npy', iid_mapping)
        np.save('../data/taobao/cid_mapping.npy', cid_mapping)
    else:
        uid_mapping = np.load('../data/taobao/uid_mapping.npy', allow_pickle=True).item()
        iid_mapping = np.load('../data/taobao/iid_mapping.npy', allow_pickle=True).item()
        cid_mapping = np.load('../data/taobao/cid_mapping.npy', allow_pickle=True).item()

    user_iid_hist = dict()
    user_cid_hist = dict()
    iid_2_cid = dict()
    multi_cid_item = set()
    all_iid = []

    n_process = 0
    with open(dataset_path, 'r') as file:
        for line in file:
            n_process += 1
            if n_process % 1000000 == 0:
                print(f'processing {n_process}')

            uid, iid, cid, btype, ts = line.split(',')
            if btype != 'pv':
                continue

            uid, iid, cid = list(map(int, [uid, iid, cid]))
            uid = uid_mapping[uid]
            iid = iid_mapping[iid]
            cid = cid_mapping[cid]

            if iid in iid_2_cid and iid_2_cid[iid] != cid:
                multi_cid_item.add(iid)
            iid_2_cid[iid] = cid

            if uid not in user_iid_hist:
                user_iid_hist[uid] = []
            user_iid_hist[uid].append(iid)

            if uid not in user_cid_hist:
                user_cid_hist[uid] = []
            user_cid_hist[uid].append(cid)

            all_iid.append(iid)

    counter = Counter(all_iid)
    print(counter.most_common(10))
    population = list(counter.keys())
    p = list(counter.values())
    for k, v in counter.most_common(10):
        assert p[population.index(k)] == v

    train_npy, val_npy, test_npy = [], [], []
    max_length = 200

    all_rnd_neg = random.choices(population, p, k=100000)
    neg_idx = 0

    same_cate_num = []
    cate_num = []

    for uid in tqdm(user_iid_hist):
        if neg_idx > 99000:
            all_rnd_neg = random.choices(population, p, k=100000)
            neg_idx = 0

        if len(user_iid_hist[uid]) < 210:
            continue

        # train sequence
        for train_i in range(3, 94, 5):
            train_npy.append([
                uid,
                user_iid_hist[uid][-train_i],
                user_cid_hist[uid][-train_i],
                [1, 0],
                pad_and_truncate(user_iid_hist[uid][:-train_i], max_length),
                pad_and_truncate(user_cid_hist[uid][:-train_i], max_length)
            ])
            for _ in range(2):
                rnd_neg = all_rnd_neg[neg_idx]
                neg_idx += 1
                while rnd_neg in user_iid_hist[uid]:
                    rnd_neg = all_rnd_neg[neg_idx]
                    neg_idx += 1
                train_npy.append([
                    uid,
                    rnd_neg,
                    iid_2_cid[rnd_neg],
                    [0, 1],
                    pad_and_truncate(user_iid_hist[uid][:-train_i], max_length),
                    pad_and_truncate(user_cid_hist[uid][:-train_i], max_length)
                ])


        # val sequence [1, T - 2] -> T - 1
        val_npy.append([
            uid,
            user_iid_hist[uid][-2],
            user_cid_hist[uid][-2],
            [1, 0],
            pad_and_truncate(user_iid_hist[uid][:-2], max_length),
            pad_and_truncate(user_cid_hist[uid][:-2], max_length)
        ])
        for _ in range(2):
            rnd_neg = all_rnd_neg[neg_idx]
            neg_idx += 1
            while rnd_neg in user_iid_hist[uid]:
                rnd_neg = all_rnd_neg[neg_idx]
                neg_idx += 1
            val_npy.append([
                uid,
                rnd_neg,
                iid_2_cid[rnd_neg],
                [0, 1],
                pad_and_truncate(user_iid_hist[uid][:-2], max_length),
                pad_and_truncate(user_cid_hist[uid][:-2], max_length)
            ])

        # test sequence [1, T - 1] -> T
        test_npy.append([
            uid,
            user_iid_hist[uid][-1],
            user_cid_hist[uid][-1],
            [1, 0],
            pad_and_truncate(user_iid_hist[uid][:-1], max_length),
            pad_and_truncate(user_cid_hist[uid][:-1], max_length)
        ])
        for _ in range(2):
            rnd_neg = all_rnd_neg[neg_idx]
            neg_idx += 1
            while rnd_neg in user_iid_hist[uid]:
                rnd_neg = all_rnd_neg[neg_idx]
                neg_idx += 1
            test_npy.append([
                uid,
                rnd_neg,
                iid_2_cid[rnd_neg],
                [0, 1],
                pad_and_truncate(user_iid_hist[uid][:-1], max_length),
                pad_and_truncate(user_cid_hist[uid][:-1], max_length),
            ])

    print('n_user:', len(uid_mapping))
    print('n_item:', len(iid_mapping))
    print('n_multi_cid_item:', len(multi_cid_item))
    print('n_cate:', len(cid_mapping))
    print('n_train:', len(train_npy))
    print('n_val:', len(val_npy))
    print('n_test:', len(test_npy))

    random.shuffle(train_npy)

    train_data_index = 1
    for i in range(0, len(train_npy), 102400):
        name = str(train_data_index)
        while len(name) < 4:
            name = "0" + name
        saved_train_array = np.asarray(train_npy[i:i + 102400], dtype=object)
        np.save('../data/taobao/train_' + name + '_' + str(len(train_npy[i:i + 102400])) + '.npy', saved_train_array)
        train_data_index += 1

    saved_val_array = np.asarray(val_npy, dtype=object)
    np.save('../data/taobao/val.npy', saved_val_array)
    saved_test_array = np.asarray(test_npy, dtype=object)
    np.save('../data/taobao/test.npy', saved_test_array)

