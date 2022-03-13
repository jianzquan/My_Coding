from torch.utils.data import Dataset, DataLoader
import numpy as np


class GenDataset(Dataset):
    def __init__(self, data):

        indexes   = [item['index'] for item in data]
        seqs      = [item['seq'] for item in data]
        seqs_id   = [item['seq_id'] for item in data]
        seqs_mask = [item['seq_mask'] for item in data]
        sent_mask = [item['sent_mask'] for item in data]
        seqs_len  = [item['seq_id_len'] for item in data]
        seqs_pos  = [item['seq_pos'] for item in data]
        asps_id   = [item['asp_id'] for item in data]
        asps_mask = [item['asp_mask'] for item in data]
        asps_len  = [item['asp_id_len'] for item in data]
        labels    = [item['label'] for item in data]

        self.indexes   = indexes
        self.seqs      = seqs
        self.seqs_id   = seqs_id
        self.seqs_mask = seqs_mask
        self.sent_mask = sent_mask
        self.seqs_len  = seqs_len
        self.seqs_pos  = seqs_pos
        self.asps_id   = asps_id
        self.asps_mask = asps_mask
        self.asps_len  = asps_len
        self.labels    = labels

    def __getitem__(self, item):
        return self.indexes[item], self.seqs[item], self.seqs_id[item], self.seqs_mask[item], self.sent_mask[item], \
        self.seqs_len[item], self.seqs_pos[item], self.asps_id[item], self.asps_mask[item], self.asps_len[item], self.labels[item]
        
    def __len__(self):
        return len(self.seqs_id)

    def _get_iter(self, batch_size, desc):
        dataLoader = DataLoader(self, batch_size, shuffle=True, collate_fn=self._colloate)
        return dataLoader

    def _colloate(_, instance_list):
        n_entity = len(instance_list[0])
        scatter_b = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(instance_list)):
            for jdx in range(0, n_entity):
                scatter_b[jdx].append(instance_list[idx][jdx])
        return scatter_b