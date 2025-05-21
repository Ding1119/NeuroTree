import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from os.path import join, exists


class BrainNetworkDataset(Dataset):
    def __init__(self, A_s, A_f_seq, X_seq, labels, ages, edge_lists):
        self.A_s = torch.FloatTensor(A_s)
        self.A_f_seq = torch.FloatTensor(A_f_seq)
        self.X_seq = torch.FloatTensor(X_seq)
        self.labels = torch.LongTensor(labels)
        self.ages = torch.FloatTensor(ages)
        self.edge_lists = edge_lists  # This is a list of lists of tuples

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.A_s[idx], 
                self.A_f_seq[idx], 
                self.X_seq[idx], 
                self.labels[idx],
                self.ages[idx],
                self.edge_lists[idx])  
    
def custom_collate(batch):
    A_s, A_f_seq, X_seq, labels, ages, edge_lists = zip(*batch)
    A_s = torch.stack(A_s, dim=0)
    A_f_seq = torch.stack(A_f_seq, dim=0)
    X_seq = torch.stack(X_seq, dim=0)
    labels = torch.stack(labels, dim=0)
    ages = torch.stack(ages, dim=0)
    edge_lists = list(edge_lists) 
    return A_s, A_f_seq, X_seq, labels, ages, edge_lists


def load_cannabis(base_dir: str):

    data_dir = join(base_dir, "data")
    parcellation_dir = join(data_dir, "compcor_nilearn_parcellation")

    df = pd.read_csv(join(base_dir, "data", "demographics.csv"))
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    raw_ts, labels, ages = [], [], []
    for _, row in df.iterrows():
        ts = pd.read_csv(
            join(parcellation_dir, f"{row.subject}_stanford_rois.csv"),
            sep="\t",
            header=None,
        ).values.T
        
        ts = padding(torch.tensor(ts))
        raw_ts.append(ts)
        labels.append(row.label)
        ages.append(row.age)

    return raw_ts, np.asarray(labels), np.asarray(ages)


def _get_paths_COBRE(phenotypes, atlas, ts_dir):
    dx_map = {"No_Known_Disorder": 0, "Schizophrenia_Strict": 2}
    phenotypes = phenotypes[phenotypes.Dx.isin(dx_map)]
    phenotypes["Dx"] = phenotypes["Dx"].map(dx_map).astype(int)
    phenotypes = phenotypes.drop_duplicates("Subject_ID")

    raw_ts, labels, ages = [], [], []
    for _, row in phenotypes.iterrows():
        f = join(ts_dir, atlas, f"{row.Subject_ID}_timeseries.txt")
        if exists(f):
            raw_ts.append(np.loadtxt(f).T)
            labels.append(row.Dx)
            ages.append(row.Age)  
    _, classes = np.unique(np.asarray(labels), return_inverse=True)
  
    return raw_ts, classes, np.asarray(ages)


def load_cobre(ts_dir: str, pheno_csv: str, atlas: str = "HarvardOxford"):
    phenotypes = pd.read_csv(pheno_csv)
    return _get_paths_COBRE(phenotypes, atlas, ts_dir)


def postprocess_timeseries(raw_ts, ages, num_timesteps):
    
    
    adj_list = []
    all_adj_matrices = []
    A_d_seq_list = []
    adj_tree_list = []
    X_seq_list = []
    edge_list_list = []
    num_nodes = raw_ts[0].shape[0]
    num_samples = len(raw_ts)

    connectivity_measure = ConnectivityMeasure(kind="correlation")

    for idx, ts in enumerate(raw_ts):
        half = ts.shape[1] // 2
        ts = torch.tensor(ts)

        adj_full = np.abs(connectivity_measure.fit_transform([ts.numpy().T])[0])
        adj_list.append(adj_full)
        adj_first_half = np.abs(connectivity_measure.fit_transform([ts[:, 0:half].numpy().T])[0])
        adj_second_half = np.abs(connectivity_measure.fit_transform([ts[:, half:].numpy().T])[0])
        
        brain_tree = tree_construction(ts.numpy())  
        adj_tree_list.append(brain_tree)
        row, col = brain_tree.nonzero()
        edge_list = list(zip(row, col))
        edge_list_list.append(edge_list)
        

        sample_adj_matrices = np.stack([adj_first_half, adj_second_half], axis=0)
        all_adj_matrices.append(sample_adj_matrices)


        X_seq_first_half = ts[:, 0:half]
        X_seq_second_half = ts[:, half:]
        # import pdb;pdb.set_trace()
        X_seq_matrices = np.stack([X_seq_first_half.numpy(), X_seq_second_half.numpy()], axis=0)
        X_seq_list.append(X_seq_matrices)


        seg_len = ts.shape[1] // 2
        first_seg = ts[:, :seg_len]
        second_seg = ts[:, seg_len:]

        A_d_matrices = np.zeros((num_timesteps, num_nodes, num_nodes))

        # t=0: calculate first_seg -> second_seg
        t = 0
        A_t = first_seg.numpy()
        A_t_plus_1 = second_seg.numpy()
        for i_node in range(num_nodes):
            for j_node in range(num_nodes):
                if A_t[i_node].mean() != 0:  # Avoid division by zero
                    A_d_matrices[t, i_node, j_node] = 0.5 * (
                        A_t_plus_1[j_node].mean() / A_t[i_node].mean() - 0.5 * ages[idx]
                    )

        # t=1: calculate second_seg 
        t = 1
        A_t = second_seg.numpy()
        for i_node in range(num_nodes):
            for j_node in range(num_nodes):
                if A_t[i_node].mean() != 0:  # Avoid division by zero
                    A_d_matrices[t, i_node, j_node] = 0.5 * (
                        A_t[j_node].mean() / A_t[i_node].mean() - 0.5 * ages[idx]
                    )
        
        A_d_seq_list.append(A_d_matrices)

    A_s = np.array(adj_list)
    A_d_seq = np.array(A_d_seq_list)
    X_seq = np.array(X_seq_list)
    edge_lists = edge_list_list

    for i in range(num_samples): 
        D_s = np.sum(A_s[i], axis=1)
        D_s_inv_sqrt = np.diag(1.0 / np.sqrt(D_s + 1e-10))
        A_s[i] = D_s_inv_sqrt @ A_s[i] @ D_s_inv_sqrt

        for t in range(num_timesteps):
            D_in = np.sum(np.abs(A_d_seq[i, t]), axis=0)
            D_out = np.sum(np.abs(A_d_seq[i, t]), axis=1)
            D_in_inv_sqrt = np.diag(1.0 / np.sqrt(D_in + 1e-10))
            D_out_inv_sqrt = np.diag(1.0 / np.sqrt(D_out + 1e-10))
            A_d_seq[i, t] = D_out_inv_sqrt @ A_d_seq[i, t] @ D_in_inv_sqrt
    # import pdb;pdb.set_trace()
    return A_s, A_d_seq, X_seq, edge_lists


def dataloader(data_type: str, num_timesteps: int):
    if data_type == "cannabis":
        raw_ts, labels, ages = load_cannabis("./datasets/cannabis")
    elif data_type == "cobre":
        raw_ts, labels, ages = load_cobre(
            "/home/jding/Music/fMRI_DTI_code/benchmark_rsfMRI_prediction/COBRE",
            "/home/jding/Music/Brain_Age_Modeling/COBRE_meta.csv",
        )
        
    else:
        raise ValueError(f"No data exist")

    A_s, A_d_seq, X_seq, edge_lists = postprocess_timeseries(raw_ts, ages, num_timesteps)
    return A_s, A_d_seq, X_seq, labels, ages, edge_lists
