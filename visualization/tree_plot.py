from collections import defaultdict
import pandas as pd
import numpy as np
import os
from visualization.plot_utils import *


def plot_brain_tree(data_type, trunks_batch, edge_lists_batch, df_roi, plot_dir='trunk_plots', save_plots=True, plot_limit=None):
    
    network_stats = {
            0: {
                'level1': defaultdict(int),
                'level2': defaultdict(int),
                'level3': defaultdict(int)
            },
            1: {
                'level1': defaultdict(int),
                'level2': defaultdict(int),
                'level3': defaultdict(int)
            }
        }
    
    if data_type == 'cannabis':
        
        if save_plots:
            plot_dir = f'brain_tree_{data_type}_visualization'
            os.makedirs(plot_dir, exist_ok=True)
            plot_count = 0

            for b in range(len(edge_lists_batch)):
                if plot_limit is not None and plot_count >= plot_limit:
                    break
                edge_list = edge_lists_batch[b]
                paths = trunks_batch[b]
                if not paths:
                    continue  
                save_path = os.path.join(plot_dir, f"graph_{plot_count + 1}.png")
                visualize_comparison(paths, save_path, df_roi )
                plot_count += 1
                if plot_limit is not None and plot_count >= plot_limit:
                    break

    elif data_type == 'cobre':
        if save_plots:
            plot_dir = f'brain_tree_{data_type}_visualization'
            os.makedirs(plot_dir, exist_ok=True)
            plot_count = 0

            for b in range(len(edge_lists_batch)):
                if plot_limit is not None and plot_count >= plot_limit:
                    break
                edge_list = edge_lists_batch[b]
                paths = trunks_batch[b]
                if not paths:
                    continue  
                save_path = os.path.join(plot_dir, f"graph_{plot_count + 1}.png")
                visualize_comparison(paths, save_path, df_roi )
                plot_count += 1
                if plot_limit is not None and plot_count >= plot_limit:
                    break

