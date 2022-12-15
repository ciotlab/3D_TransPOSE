import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from einops import rearrange
import torch
import logging

area_min_xyz = np.array([-1.5, 0.5, 0.0])
area_max_xyz = np.array([1.5, 3.5, 2.5])
area_size_xyz = area_max_xyz - area_min_xyz
box_margin = np.array([0.6, 0.6, 0])  # box margin from the skeleton (meter)


class RadarDataset(Dataset):
    def __init__(self, min_seq_idx=64, num_stacked_seqs=1):
        file_dir = Path(__file__).parents[1].resolve() / 'data'
        self._label_file_dir = file_dir / 'label_raw'
        self._radar_file_dir = file_dir / 'radar_declutter'
        index_df_file = file_dir / 'index_dataframe.csv'
        self._index = pd.read_csv(index_df_file)
        self._min_seq_idx = min_seq_idx
        self._num_stacked_seqs = num_stacked_seqs
        self._min_stacked_seq_idx = self._min_seq_idx + self._num_stacked_seqs - 1
        min_seq_query = '(sequence >= @self._min_stacked_seq_idx)'
        self._index_set = self._index[['location', 'num_person', 'session', 'sequence']].query(min_seq_query)
        self._sel_index = None

    def select(self, q):
        if q == '':
            self._sel_index = self._index_set
        else:
            self._sel_index = self._index_set.query(q)

    def add(self, q):
        tmp = self._index_set.query(q)
        self._sel_index = pd.concat([self._sel_index, tmp], axis=0, ignore_index=True)

    def set_minus(self, q):
        q = 'not (' + q + ')'
        self._sel_index = self._sel_index.query(q)

    def intersect(self, q):
        self._sel_index = self._sel_index.query(q)

    @staticmethod
    def get_normalized_box_and_keypoint(label):
        keypoint = np.stack(np.split(label, label.shape[0] // 21, axis=0))  # (person, point, xyz)
        box_min = keypoint.min(axis=1, keepdims=True, initial=np.inf) - box_margin/2.0
        box_max = keypoint.max(axis=1, keepdims=True, initial=-np.inf) + box_margin/2.0
        # get normalized box
        box_center = (box_min + box_max) / 2
        box_size = box_max - box_min
        norm_box_center = (box_center - area_min_xyz) / area_size_xyz
        norm_box_size = box_size / area_size_xyz
        norm_box = np.concatenate((norm_box_center, norm_box_size), axis=1)
        # get normalized keypoint
        norm_keypoint = (keypoint - box_min) / box_size
        return norm_box, norm_keypoint

    def __len__(self):
        return self._sel_index.shape[0]

    def __getitem__(self, idx):
        x = self._sel_index.iloc[idx]
        last_seq = x['sequence']
        first_seq = last_seq - self._num_stacked_seqs + 1
        data = self._index[(self._index['location'] == x['location']) & (self._index['num_person'] == x['num_person'])
                           & (self._index['session'] == x['session']) & (self._index['sequence'] >= first_seq)
                           & (self._index['sequence'] <= last_seq)]
        radar = []
        for d in data.iterrows():
            radar.append(np.load(d[1]['radar']))
        radar = np.stack(radar)
        label_file = data.iloc[-1]['label']
        label = np.load(label_file)
        norm_box, norm_keypoint = self.get_normalized_box_and_keypoint(label)
        label = (norm_box, norm_keypoint, label_file)
        return radar, label


def collate_fn(batch):
    radar, label = zip(*batch)
    radar = np.stack(radar)
    radar = rearrange(radar, 'b f t r s -> b f (t r) s')
    radar = torch.tensor(radar).float()
    boxes, keypoints, label_files = zip(*label)
    label = {'boxes': list(boxes), 'keypoints': list(keypoints), 'label_files': list(label_files)}
    return radar, label


def get_dataset_and_dataloader(scope_query, test_query, batch_size, num_workers=0, num_stacked_seqs=4, mode='train'):
    dataset = RadarDataset(min_seq_idx=64, num_stacked_seqs=num_stacked_seqs)
    dataset.select(scope_query)
    if mode == 'train':
        dataset.set_minus(test_query)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                collate_fn=collate_fn)
    else:
        dataset.intersect(test_query)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=collate_fn)
    return dataloader, dataset


def get_dataset_and_dataloader_num_person(location, num_person, test_session, batch_size, num_workers=0,
                                          num_stacked_seqs=4, mode='train'):
    scope_query = f"(location=='{location}') and (num_person=={num_person})"
    test_query = f"(session=={test_session})"
    dataloader, dataset = get_dataset_and_dataloader(scope_query, test_query, batch_size=batch_size,
                                                     num_workers=num_workers, num_stacked_seqs=num_stacked_seqs, mode=mode)
    return dataloader, dataset


def get_dataset_and_dataloader_all(batch_size, num_workers=0, num_stacked_seqs=4, mode='train'):
    scope_query = ""
    test_query = "(session==1)"
    dataloader, dataset = get_dataset_and_dataloader(scope_query, test_query, batch_size=batch_size,
                                                     num_workers=num_workers, num_stacked_seqs=num_stacked_seqs, mode=mode)
    return dataloader, dataset


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    # dataloader, dataset = get_dataset_and_dataloader_num_person(location='B', num_person=1, test_session=1,
    #                                                             batch_size=16, num_workers=0, num_stacked_seqs=6,
    #                                                             mode='test')
    dataloader, dataset = get_dataset_and_dataloader_all(batch_size=16, num_workers=0, num_stacked_seqs=2,
                                                        mode='test')
    data_it = iter(dataloader)
    radar, label = next(data_it)
    a = 1

