import torch
from nilearn.connectome import ConnectivityMeasure
from scipy.sparse.csgraph import minimum_spanning_tree

def tree_construction(fmri_timeseries):
    """
    adapted from Huang, Zhongyu, et al. "Identifying the hierarchical emotional areas in the human brain through information fusion." Information Fusion 113 (2025): 102613.

    https://github.com/zhongyu1998/HEmoN
    """
    

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



