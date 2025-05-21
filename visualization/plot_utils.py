import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx
import itertools
from scipy import stats

def get_nature_colors():
    """
    return Nature Journal style colors for networks
    """
    # return {
    #     'Default': '#E64B35',          
    #     'Frontoparietal': '#4DBBD5',   
    #     'Somatomotor': '#00A087',      
    #     'Dorsal Attention': '#3C5488', 
    #     'Subcortical': '#F39B7F',      
    #     'Others': '#8491B4'           
    # }
    # return {
    # 'Default': '#E64B35',          
    # 'Frontoparietal': '#4DBBD5',   
    # 'Somatomotor': '#00A087',      
    # 'Dorsal Attention': '#7E6148', 
    # 'Subcortical': '#F39B7F',      
    # 'Others': '#8491B4'           
    # }
    return {
    'Default': '#E64B35',          
    'Frontoparietal': '#4DBBD5',   
    'Somatomotor': '#00A087',      
    'Dorsal Attention': '#FFD700',
    'Subcortical': '#8B4513',      
    'Visual': '#9932CC',
    'Others': '#808080'            
        }


def create_original_graph(edge_list, df):
    G = nx.Graph()
    G.add_edges_from(edge_list)
    
    # Create a mapping of node index to network color
    color_map = df.set_index('parcel_ind')['network_color'].to_dict()
    
    # Add color attribute to each node
    nx.set_node_attributes(G, {node: color_map.get(node, '#808080') for node in G.nodes()}, 'color')
    
    return G


def visualize_comparison(paths, save_path, df):
   

    G_trunk      = nx.Graph()
    level_colors = ['#DC143C', '#FFC0CB', '#FFE4B5', '#FFFAF0']   # each level color
    edge_colors  = []
    edges        = []

    for level, path_list in enumerate(paths):
        for path in path_list:
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                edges.append(edge)
                edge_colors.append(level_colors[level % len(level_colors)])
                G_trunk.add_edge(*edge)

    pos_trunk          = nx.kamada_kawai_layout(G_trunk)
    color_map          = df.set_index('parcel_ind')['network_color'].to_dict()
    trunk_node_colors  = [color_map.get(node, '#808080') for node in G_trunk.nodes()]


    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Trunk Paths Structure", pad=20, size=18, weight='bold')

    for edge, c in zip(edges, edge_colors):
        nx.draw_networkx_edges(G_trunk, pos_trunk,
                               edgelist=[edge],
                               edge_color=c,
                               width=2,
                               ax=ax)

    nx.draw_networkx_nodes(G_trunk, pos_trunk,
                           node_color=trunk_node_colors,
                           node_size=300,
                           ax=ax)

    level_legend = [
        mpatches.Patch(color=level_colors[i % len(level_colors)],
                       label=f'Level {i + 1}')
        for i in range(len(paths))
    ]
    ax.legend(handles=level_legend,
              loc='center left',
              bbox_to_anchor=(1, 0.5),
              fontsize=18)

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def calculate_significance(network_stats, label, network_order):
    """
     calculates the significance of differences between network connectivity counts
     for different levels using Mann-Whitney U test.
    """
    
    significance_data = {level: {} for level in ['level1', 'level2', 'level3']}
    
    for level in ['level1', 'level2', 'level3']:
        
        level_data = network_stats[label][level]
        
        
        for net1, net2 in itertools.combinations(network_order, 2):
           
            count1 = level_data.get(net1, 0)
            count2 = level_data.get(net2, 0)
            
            
            sample1 = [1] * count1 + [0] * (max(count1, count2) - count1)
            sample2 = [1] * count2 + [0] * (max(count1, count2) - count2)
            
            try:
                
                _, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
            except ValueError:
                
                p_value = 1.0
            
            
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = 'ns'
                
            significance_data[level][(net1, net2)] = {'p_value': p_value, 'significance': sig}
    
    return significance_data



def create_combined_network_plot(network_stats, label, nature_colors):
    """
    create a combined network plot for three levels of connectivity
    """
    
    all_networks = set()
    for level in ['level1', 'level2', 'level3']:
        all_networks.update(network_stats[label][level].keys())
    
    
    df_list = []
    for level in ['level1', 'level2', 'level3']:
        data = network_stats[label][level]
        df = pd.DataFrame({
            'Network': list(all_networks),
            'Count': [data.get(net, 0) for net in all_networks],
            'Level': [f'Level {level[-1]}'] * len(all_networks)
        })
        df_list.append(df)
    
    df = pd.concat(df_list)
    
    
    network_totals = df.groupby('Network')['Count'].sum().sort_values(ascending=True)
    network_order = network_totals.index.tolist()
    
    
    significance_data = calculate_significance(network_stats, label, network_order)
    
    
    plt.figure(figsize=(12, max(8, len(network_order)*0.8)))
    
    
    bar_width = 0.25
    r1 = np.arange(len(network_order))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    
    colors = [nature_colors.get(network, nature_colors['Others']) for network in network_order]
    
    
    bar_ends = {level: {} for level in ['level1', 'level2', 'level3']}
    
   
    for i, level in enumerate(['level1', 'level2', 'level3']):
        counts = [network_stats[label][level].get(net, 0) for net in network_order]
        positions = [r1, r2, r3][i]
        alpha = [1.0, 0.7, 0.4][i]
        
        bars = plt.barh(positions, counts, bar_width, 
                       color=colors, alpha=alpha, 
                       label=f'Level {level[-1]}')
        
        
        for idx, (bar, net) in enumerate(zip(bars, network_order)):
            bar_ends[level][net] = bar.get_width()
        
        
        for bar_idx, bar in enumerate(bars):
            width = bar.get_width()
            if width > 0:
                plt.text(width, bar.get_y() + bar.get_height()/2,
                        f'{int(width)}',
                        ha='left', va='center', fontsize=12)

    max_x = max([max(ends.values()) for ends in bar_ends.values()])
    
    line_spacing = max_x * 0.02
    
    # add each level's significance
    for level in ['level1', 'level2', 'level3']:
        positions = {'level1': r1, 'level2': r2, 'level3': r3}[level]
        
        current_max_x = max_x + line_spacing
        
        # only connect networks with significant differences
        for idx in range(len(network_order)-1):
            net1 = network_order[idx]
            net2 = network_order[idx+1]
            sig_data = significance_data[level].get((net1, net2))
            
            if sig_data and sig_data['significance'] != 'ns':
                y1 = positions[idx] + bar_width/2
                y2 = positions[idx+1] + bar_width/2
                
                
                x_start = max(bar_ends[level][net1], bar_ends[level][net2])
                
                x_extend = x_start + line_spacing
                
                
                plt.plot([x_start, x_extend, x_extend, x_start],
                        [y1, y1, y2, y2],
                        color='black', linewidth=1)
                
                
                plt.text(x_extend + line_spacing*0.5, (y1 + y2)/2,
                        sig_data['significance'],
                        ha='center', va='center')
                
                
                current_max_x = x_extend + line_spacing

    extra_space = line_spacing * 4  
    x_lim_right = max_x + extra_space
    plt.xlim(right=x_lim_right)
    
    
    plt.xlabel('Number of connectivity', labelpad=13, fontsize=18, weight='bold')
    plt.yticks(r2, network_order, fontsize=18, weight='bold')
    plt.ylabel('Networks', fontsize=18, weight='bold')
    plt.xticks(fontsize=18, weight='bold')
    
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
   
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    
    plt.subplots_adjust(right=0.9)  
    
    return plt.gcf()