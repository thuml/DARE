import os
import numpy as np


class AnalysisDataset:  # only used for analysis
    def __init__(self, train_dir_path, test_file_path):
        assert "{}" in train_dir_path
        dir_name = os.path.dirname(train_dir_path)
        base_name = os.path.basename(train_dir_path)
        filter_name = base_name.split("{")[0]
        files = os.listdir(dir_name)
        self.training_files = sorted([file for file in files if filter_name in file])
        self.train_data_number = []
        for i, file in enumerate(self.training_files):
            index_and_num = file.split(filter_name)[1].split(".npy")[0].split("_")
            index = index_and_num[0]
            num = index_and_num[1]
            assert int(index) == i + 1
            self.train_data_number.append(int(num))
        for i in range(len(self.training_files)):
            self.training_files[i] = os.path.join(dir_name, self.training_files[i])
        self.train_files_num = len(self.training_files)
        self.train_data_size = sum(self.train_data_number)
        self.train_data = np.load(self.training_files[0], allow_pickle=True)
        self.train_file_index = 0
        self.train_i = 0

        if test_file_path is not None and test_file_path != "None":
            self.test_data = np.load(test_file_path, allow_pickle=True)
            self.test_data_num = len(self.test_data)
            self.test_i = 0
        else:
            self.test_data = None
            self.test_data_num = 0

    def __len__(self):
        return self.train_data_size + self.test_data_num

    def __iter__(self):
        return self

    def __next__(self):
        if self.train_file_index >= self.train_files_num:
            if self.test_data is None:
                raise StopIteration
            if self.test_i >= self.test_data_num:
                raise StopIteration
            else:
                aim_data = self.test_data[self.test_i]
                self.test_i += 1
                return aim_data
        else:
            aim_data = self.train_data[self.train_i]
            self.train_i += 1
            if self.train_i >= self.train_data_number[self.train_file_index]:
                self.train_i = 0
                self.train_file_index += 1
                if self.train_file_index < self.train_files_num:
                    self.train_data = np.load(self.training_files[self.train_file_index], allow_pickle=True)
            return aim_data


