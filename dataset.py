import warnings
import os

import numpy as np


class SeqRecDataset:

    def __init__(self, dataset_path, batch_size=128, max_length=100, apply_hard_search=False):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.apply_hard_search = apply_hard_search
        if "{}" in self.dataset_path:
            self.one_file = False
            dir_name = os.path.dirname(self.dataset_path)
            base_name = os.path.basename(self.dataset_path)
            filter_name = base_name.split("{")[0]
            files = os.listdir(dir_name)
            self.training_files = sorted([file for file in files if filter_name in file])
            self.data_number = []
            for i, file in enumerate(self.training_files):
                index_and_num = file.split(filter_name)[1].split(".npy")[0].split("_")
                index = index_and_num[0]
                num = index_and_num[1]
                assert int(index) == i+1
                self.data_number.append(int(num))
            for i in range(len(self.training_files)):
                self.training_files[i] = os.path.join(dir_name, self.training_files[i])
            self.data_size = sum(self.data_number)
            self.data = np.load(self.training_files[0], allow_pickle=True)
            self.file_index = 0
            self.i = 0
        else:
            self.one_file = True
            self.data = np.load(self.dataset_path, allow_pickle=True)
            self.data_size = len(self.data)
            self.i = 0
            self.epoch_size = self.data_size // self.batch_size


    def __iter__(self):
        return self

    def __next__(self):
        if self.one_file == True:
            if self.i == self.epoch_size:
                raise StopIteration
            batch = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size, len(self.data))]
            self.i += 1
        else:
            if self.file_index == len(self.training_files):
                raise StopIteration
            batch = self.data[self.i : self.i+self.batch_size]
            self.i += self.batch_size
            if self.i >= len(self.data):
                self.i = 0
                self.file_index += 1
                if self.file_index < len(self.training_files):
                    self.data = np.load(self.training_files[self.file_index], allow_pickle=True)

        uid, iid, cid, target, hist_iid, hist_cid = [], [], [], [], [], []
        for x in batch:
            uid.append(x[0])
            iid.append(x[1])
            cid.append(x[2])
            target.append(x[3])
            hist_iid.append(x[4][-self.max_length:])
            hist_cid.append(x[5][-self.max_length:])

        batch_size = len(uid)

        uid, iid, cid, target, hist_iid, hist_cid = list(map(np.array, [uid, iid, cid, target, hist_iid, hist_cid]))
        mask = np.greater(hist_iid, 0).astype(np.float32)
        if self.apply_hard_search:
            mask = (hist_cid == cid[:, None]).astype(np.float32) * mask

        tid = np.zeros(iid.shape, dtype=np.int32)
        # fny_tocheck: 编号，离目标越近编号越小
        hist_tid = np.array([[self.max_length - x for x in range(self.max_length)]])
        hist_tid = np.repeat(hist_tid, batch_size, axis=0)

        return {
            'uid': uid,
            'iid': iid,
            'cid': cid,
            'tid': tid,
            'hist_iid': hist_iid,
            'hist_cid': hist_cid,
            'hist_tid': hist_tid,
            'mask': mask,
            'target': target
        }

    def reset(self):
        if not self.one_file:
            self.file_index = 0
            self.data = np.load(self.training_files[0], allow_pickle=True)
        self.i = 0

    def shuffle(self):
        if not self.one_file:
            warnings.warn("shuffle failed!")
            return
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        self.data = self.data[indices]

