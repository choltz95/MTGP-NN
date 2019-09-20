import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy import stats
import numpy as np
import pickle
from random import shuffle
from pathlib import Path

class Dataset:
    def __init__(self):
        self.groups = {}
        for group_name in ('train', 'val', 'test'):
            self.groups[group_name] = {
                "pid_list": [],
                "vital_features_list": [],
                "baseline_features_list": [],
                "lab_features_list": [],
                "ignored_features_list": [],
                "labels_list": []
            }
        
    @classmethod
    def generate_pid_split(cls, data, split):
        # split given as int
        self = cls()
        pids = data.pid.tolist()
        shuffle(pids)
        spl = int(split * len(pids))
        tst_pids = pids[:spl]
        val_pids = pids[spl:2*spl]
        tr_pids = pids[2*spl:]
        
        pid_grouped_data = data.groupby('pid')
        for pid, group in tqdm(pid_grouped_data, desc='splitting data...'):
            if pid in tr_pids:
                self.groups['train']['pid_list'].append(pid)
                self.groups['train']['vital_features_list'].append(group[vital_covars].reset_index())
                self.groups['train']['baseline_features_list'].append(group[baseline_covars].reset_index())
                self.groups['train']['lab_features_list'].append(group[lab_covars].reset_index())
                self.groups['train']['ignored_features_list'].append(group[ignored_covars].reset_index())
                self.groups['train']['labels_list'].append(group[label])
            elif pid in val_pids:
                self.groups['val']['pid_list'].append(pid)
                self.groups['val']['vital_features_list'].append(group[vital_covars].reset_index())
                self.groups['val']['baseline_features_list'].append(group[baseline_covars].reset_index())
                self.groups['val']['lab_features_list'].append(group[lab_covars].reset_index())
                self.groups['val']['ignored_features_list'].append(group[ignored_covars].reset_index())
                self.groups['val']['labels_list'].append(group[label])
            elif pid in tst_pids:
                self.groups['test']['pid_list'].append(pid)
                self.groups['test']['vital_features_list'].append(group[vital_covars].reset_index())
                self.groups['test']['baseline_features_list'].append(group[baseline_covars].reset_index())
                self.groups['test']['lab_features_list'].append(group[lab_covars].reset_index())
                self.groups['test']['ignored_features_list'].append(group[ignored_covars].reset_index())
                self.groups['test']['labels_list'].append(group[label])
        return self
    
    @classmethod
    def load(cls, filepath):
        filepath = Path(filepath)
        with filepath.open('rb') as f:
            return pickle.load(f)
        
    def save(self, filepath):
        filepath = Path(filepath)
        if filepath.exists():
            raise FileExistsError(f"cannot write dataset to '{filepath}'; file exists")
        
        with filepath.open('wb') as f:
            pickle.dump(self, f)

