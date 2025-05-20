import pandas as pd
import numpy as np
import torch
from nilearn.connectome import ConnectivityMeasure
from scipy.sparse.csgraph import minimum_spanning_tree
from torch.utils.data import Dataset

def tree_construction(fmri_timeseries):

    fmri_timeseries = fmri_timeseries.T

    correlation_measure = ConnectivityMeasure(kind="correlation")
    correlation_matrix = correlation_measure.fit_transform([fmri_timeseries])[0]

    spanning_tree = minimum_spanning_tree(1 - correlation_matrix)

    brain_tree = (spanning_tree + spanning_tree.T) > 0
    brain_tree = brain_tree.toarray().astype(float)

    return brain_tree

def padding(data):
    n_channels, n_length = data.size()

    target_length = 810

    if n_length < target_length:
        pad_length = target_length - n_length
  
        mean_value = torch.mean(data, dim=1, keepdim=True)  

        padded_data = torch.cat([data, mean_value.expand(n_channels, pad_length)], dim=1)
    else:
        padded_data = data  

    return padded_data

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


def dataloader(data_type, num_timesteps):
  
    if data_type == 'cannabis':
        base_dir     = '/home/jding/Documents/datasets/Cannabis/'
        data_dir     = base_dir + "data"
        analysis_dir = base_dir + '/analysis'
        feat_mat_dir = analysis_dir + '/feature_matrices'
        parcellation_dir = f'{data_dir}/compcor_nilearn_parcellation/'
        splits_df = pd.read_csv('/home/jding/Documents/cannabis-classifier/new_df.csv')
        splits_df['age'] = pd.to_numeric(splits_df['age'], errors='coerce')
        ages = np.array(splits_df['age'])
        labels = np.array(splits_df['label'])

        raw = []
        for sub_name in splits_df['subject']:
            full_ts = pd.read_csv(f'{parcellation_dir}/{sub_name}_stanford_rois.csv', sep='\t', header=None).values.T
            full_ts = padding(torch.tensor(full_ts))
            raw.append(full_ts)

        adj_list = []
        all_adj_matrices = []
        A_d_seq_list = []
        adj_tree_list = []
        X_seq_list = []
        edge_list_list = []
        num_nodes = raw[0].shape[0]
        num_samples = len(raw)

        connectivity_measure = ConnectivityMeasure(kind="correlation")

        for i, (tseries, sub_name) in enumerate(zip(raw, splits_df['subject'])):
            half = tseries.shape[1] // 2
            tseries = torch.tensor(tseries)
            tseries = padding(tseries)

            adj_full = np.abs(connectivity_measure.fit_transform([tseries.numpy().T])[0])
            adj_list.append(adj_full)
            adj_first_half = np.abs(connectivity_measure.fit_transform([tseries[:, 0:half].numpy().T])[0])
            adj_second_half = np.abs(connectivity_measure.fit_transform([tseries[:, half:].numpy().T])[0])
            
            brain_tree = tree_construction(tseries.numpy())  
            adj_tree_list.append(brain_tree)
            row, col = brain_tree.nonzero()
            edge_list = list(zip(row, col))
            edge_list_list.append(edge_list)
            

            sample_adj_matrices = np.stack([adj_first_half, adj_second_half], axis=0)
            all_adj_matrices.append(sample_adj_matrices)


            X_seq_first_half = tseries[:, 0:half]
            X_seq_second_half = tseries[:, half:]
            X_seq_matrices = np.stack([X_seq_first_half.numpy(), X_seq_second_half.numpy()], axis=0)
            X_seq_list.append(X_seq_matrices)


            seg_len = tseries.shape[1] // 2
            first_seg = tseries[:, :seg_len]
            second_seg = tseries[:, seg_len:]
 
            A_d_matrices = np.zeros((num_timesteps, num_nodes, num_nodes))

            # t=0: calculate first_seg -> second_seg
            t = 0
            A_t = first_seg.numpy()
            A_t_plus_1 = second_seg.numpy()
            for i_node in range(num_nodes):
                for j_node in range(num_nodes):
                    if A_t[i_node].mean() != 0:  # Avoid division by zero
                        A_d_matrices[t, i_node, j_node] = 0.5 * (
                            A_t_plus_1[j_node].mean() / A_t[i_node].mean() - 0.5 * ages[i]
                        )

            # t=1: calculate second_seg 
            t = 1
            A_t = second_seg.numpy()
            for i_node in range(num_nodes):
                for j_node in range(num_nodes):
                    if A_t[i_node].mean() != 0:  # Avoid division by zero
                        A_d_matrices[t, i_node, j_node] = 0.5 * (
                            A_t[j_node].mean() / A_t[i_node].mean() - 0.5 * ages[i]
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
        
        return A_s, A_d_seq, X_seq, labels, ages, edge_lists
        

    elif data_type == 'cobre':
        pass