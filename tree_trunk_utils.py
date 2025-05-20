import networkx as nx

def find_trunk_gnn_batch_with_fc_strength(node_scores_batch, edge_lists_batch, fc_strength_batch, max_level, S=2, alpha=0.1):
    """Find trunk paths using high-order path information based on GPC (Graph Path Convolution).
    
    Args:
        node_scores_batch (torch.Tensor): [batch_size, num_nodes] node scores tensor
        edge_lists_batch (list): List of edge lists for each graph
        fc_strength_batch (torch.Tensor): [batch_size, num_nodes] functional connectivity strengths
        max_level (int): Maximum trunk levels to find
        S (int): Highest order of paths to consider (as per Formula 1)
        alpha (float): Weight balance between node scores and edge weights
    
    Returns:
        list: Batch of trunk paths for each graph
    """
    trunk_list_batch = []
    batch_size = node_scores_batch.size(0)
    num_nodes = node_scores_batch.size(1)

    def calculate_path_gpc(G, path, fc_strength, S):
   
        path_weight = 0
        
        # For each pair of nodes in path
        for idx in range(len(path) - 1):
            i, j = path[idx], path[idx + 1]
            
            # Direct connection (0-order path)
            P0 = fc_strength[i] + fc_strength[j]  # Sum of node strengths
            path_weight += P0.item()  # Convert to scalar
            
            # Higher order paths through other nodes
            for s in range(1, S+1):
                # Find all s-order paths between i and j
                s_order_paths = []
                
                def find_s_order_paths(current_path, length):
                    if length == s + 1:
                        if current_path[0] == i and current_path[-1] == j:
                            s_order_paths.append(current_path)
                        return
                    for node in G.neighbors(current_path[-1]):
                        if node not in current_path:
                            find_s_order_paths(current_path + [node], length + 1)

                find_s_order_paths([i], 1)
                
                # Calculate contribution of s-order paths
                for path_k in s_order_paths:
                    # Sum of node strengths along path
                    Ps_k = sum(fc_strength[node].item() for node in path_k)
                    path_weight += Ps_k
        
        return path_weight

    for b in range(batch_size):
        node_scores = node_scores_batch[b]  # [num_nodes]
        edge_list = edge_lists_batch[b]
        fc_strength = fc_strength_batch[b]  # [num_nodes]
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        for u, v in edge_list:
            # Add edge with weight based on node strengths
            weight = (fc_strength[u] + fc_strength[v]).item()
            G.add_edge(u, v, weight=weight)
            
        trunk_list = []
        level = 0
        
        while G.nodes and level < max_level:
            level += 1
            level_list = []
            
            # Map node scores
            scores = {node: node_scores[node].item() for node in G.nodes}
            
            # Process each connected component
            for component in nx.connected_components(G):
                if len(component) < 2:
                    continue
                    
                subgraph = G.subgraph(component).copy()
                component_scores = {node: scores[node] for node in component}
                
                # Find highest scoring start node
                start_node = max(component_scores.items(), key=lambda x: x[1])[0]
                
                # Calculate weighted paths using GPC
                def weighted_path_length(node1, node2):
                    path = nx.shortest_path(subgraph, node1, node2)
                    gpc_weight = calculate_path_gpc(subgraph, path, fc_strength, S)
                    # print(gpc_weight)
                    # import pdb;pdb.set_trace()
                    path_score = sum(scores[n] for n in path)
                    
                    return alpha * path_score + (1 - alpha) * gpc_weight
                
                    # return path_score + gpc_weight
                
                # Find end node maximizing weighted path length
                distances = {node: weighted_path_length(start_node, node) 
                           for node in component}
                # import pdb;pdb.set_trace()
                end_node = max(distances.items(), key=lambda x: x[1])[0]
                
                # Extract shortest path
                path = nx.shortest_path(subgraph, start_node, end_node)
                level_list.append(path)
                
                # Remove path edges
                path_edges = list(zip(path[:-1], path[1:]))
                G.remove_edges_from(path_edges)
            
            # Remove isolated nodes
            G.remove_nodes_from(list(nx.isolates(G)))
            
            if level_list:
                trunk_list.append(level_list)
                
        trunk_list_batch.append(trunk_list)
        
    return trunk_list_batch