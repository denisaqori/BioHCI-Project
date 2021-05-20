import torch
import numpy as np

from BioHCI.data_augmentation.vae_generator import VAE_Generator
from BioHCI.helpers import utilities as utils
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

class VAE_Categories():
    def __init__(self, train_dataset, val_dataset, category_map, learning_def, batch_size=128, seed=1):
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if learning_def.use_cuda else "cpu")
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if learning_def.use_cuda else {}

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.category_map = category_map
        self.batch_size = batch_size

        self.train_category_set = self.get_category_set(train_dataset)
        self.val_category_set = self.get_category_set(val_dataset)
        assert set(self.val_category_set).issubset(set(self.train_category_set)), \
            "All validation categories need to be part of the train categories."

        self.generate_all_vae()
        print("")

    @staticmethod
    def get_category_set(subj_dataset):
        # create a list of all classes uniquely represented
        cat_list = []
        for subj_name, subj in subj_dataset.items():
            for i, cat in enumerate(subj.categories):
                cat_list.append(subj.categories[i])
        cat_set = list(set(cat_list))
        return cat_set

    @staticmethod
    def get_data_by_category(subj_dataset, category_set):
        # create a category dictionary, where a category key is related to a list of
        # all of its samples in all the subjects
        cat_dict = {}
        for i, cat in enumerate(category_set):
            cat_dict[cat] = []
            for subj_name, subj in subj_dataset.items():
                for j, subj_cat in enumerate(subj.categories):
                    if cat == subj_cat:
                        cat_dict[cat].append(subj.data[j])

        return cat_dict

    def get_loader(self, dataset, cat):
        np_data = np.asarray(dataset, dtype=np.float32)
        data = torch.from_numpy(np_data)

        # convert categories from string to integer
        labels = utils.convert_categories(self.category_map, cat)
        labels = torch.from_numpy(labels)

        # the tensor_dataset is a tuple of TensorDataset type, containing a tensor with data (train or val),
        # and one with labels (train or val respectively)
        tensor_dataset = TensorDataset(data, labels)
        data_loader = DataLoader(tensor_dataset, batch_size=self.batch_size,
                                 num_workers=1, shuffle=False, pin_memory=True)
        return data_loader

    def generate_all_vae(self):
        train_cat_dict = self.get_data_by_category(self.train_dataset, self.train_category_set)
        val_cat_dict = self.get_data_by_category(self.val_dataset, self.val_category_set)

        for cat, train_data in train_cat_dict.items():
            train_cat = [cat for _ in range(0, len(train_data))]
            train_loader = self.get_loader(train_data, train_cat)

            val_data = val_cat_dict[cat]
            val_cat = [cat for _ in range(0, len(val_data))]
            val_loader = self.get_loader(val_data, val_cat)

            vae_gen = VAE_Generator(train_loader, val_loader, self.device, self.batch_size, name=str(cat))
            vae_gen.perform_cv()

        print("")

