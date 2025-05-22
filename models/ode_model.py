import torch
import torch.nn as nn
import torch.nn.functional as F

class AGE_GraphEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes):
        super(AGE_GraphEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        self.W = nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
        self.gamma = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.lambda_param = nn.Parameter(torch.FloatTensor(1))
        self.beta = nn.Parameter(torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.gamma)
        nn.init.constant_(self.lambda_param, 0.5)
        nn.init.constant_(self.beta, 0.1)

    def forward(self, A_s, A_d, X, age):
        """
        Args:
            A_s: [batch_size, num_nodes, num_nodes]
            A_d: [batch_size, num_nodes, num_nodes]
            X: [batch_size, num_nodes, input_dim]
            age: [batch_size, 1] or scalar representing age
        """
        # Constrain lambda and beta to [0,1]
        lambda_val = torch.sigmoid(self.lambda_param)
        beta_val = torch.sigmoid(self.beta)

        # Combine forward and backward effective connectivity
        A_combined = lambda_val * A_d + (1 - lambda_val) * A_d.transpose(-2, -1)
        # A_combined = torch.linalg.matrix_power(A_combined, 3)

        # Apply structural constraint and learnable edge weights
        gamma_expanded = self.gamma.unsqueeze(0).expand(A_s.size(0), -1, -1)
        A_weighted = gamma_expanded * A_s * A_combined

        # Incorporate age effect
        age_expanded = age.unsqueeze(-1).unsqueeze(-1) if len(age.shape) == 1 else age.view(-1, 1, 1)
        age_effect = beta_val * age_expanded * X
        # age_effect = beta_val  * X
        # import pdb;pdb.set_trace()
        # Graph convolution
   
        # Z = torch.matmul(A_weighted, torch.matmul(X * age_effect, self.W))
   
        Z = torch.matmul(A_weighted, torch.matmul(X * age_effect, self.W))
        # Z = torch.matmul(A_weighted, torch.matmul(X , self.W))
        return F.relu(Z)
    

class NeuroODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_timesteps, num_nodes):
        super(NeuroODE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.num_nodes = num_nodes
        
        # Graph embedding layer
        self.graph_embedding = AGE_GraphEmbedding(input_dim, hidden_dim, num_nodes)
        
        # Classification layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.mlp_age = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Node score prediction
        self.node_score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # FC strength projection layers
        self.fc_projector = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes)
        )
        
        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, A_s, A_d_seq, X_seq, age, train_stage=None):
        batch_size = A_s.size(0)
        
        # Initialize embeddings
        Z = torch.zeros(batch_size, self.num_nodes, self.hidden_dim).to(A_s.device)
        fc_strength = torch.zeros(batch_size, self.num_nodes).to(A_s.device)
        
        contrastive_loss = 0
        
        for t in range(self.num_timesteps):
            # Current functional connectivity matrix
            curr_fc = A_d_seq[:, t]  # [batch_size, num_nodes, num_nodes]
            
            # Calculate FC strength for current timestep
            curr_fc_strength = curr_fc.abs().mean(dim=2)  # [batch_size, num_nodes]
            # import pdb;pdb.set_trace()
            # Project FC strength for contrastive learning
            projected_fc = self.fc_projector(curr_fc_strength)  # [batch_size, num_nodes]
            
            # Generate positive and negative pairs
            # Positive: nodes with high FC strength
            # Negative: nodes with low FC strength
            mean_fc = curr_fc_strength.mean(dim=1, keepdim=True)
            positive_mask = (curr_fc_strength >= mean_fc).float()
            negative_mask = (curr_fc_strength < mean_fc).float()
            
            # Calculate contrastive loss
            similarity_matrix = F.cosine_similarity(
                projected_fc.unsqueeze(1),
                projected_fc.unsqueeze(2),
                dim=-1
            ) / self.temperature
            
            # Positive pair loss
            positive_pairs = similarity_matrix * positive_mask
            positive_loss = -torch.log(
                torch.exp(positive_pairs) / 
                (torch.exp(similarity_matrix).sum(dim=-1, keepdim=True) + 1e-6)
            ).mean()
            
            # Negative pair loss
            negative_pairs = similarity_matrix * negative_mask
            negative_loss = -torch.log(
                1 - torch.exp(negative_pairs) / 
                (torch.exp(similarity_matrix).sum(dim=-1, keepdim=True) + 1e-6)
            ).mean()
            
            # Combine losses
            contrastive_loss += positive_loss + negative_loss
            contrastive_loss = contrastive_loss / 2
            
            # Update fc_strength using contrastive learning
            fc_strength += F.softmax(projected_fc, dim=-1)
            
            # Update node embeddings
            Z_new = self.graph_embedding(A_s, A_d_seq[:, t], X_seq[:, t], age)
            Z = Z + Z_new
 
        # Average and normalize fc_strength
        # fc_strength = fc_strength / self.num_timesteps
        fc_strength = self.update_fc_strength(fc_strength)
        
        # Global average pooling for graph-level prediction
        Z_graph = torch.mean(Z, dim=1)  # [batch_size, hidden_dim]
        out = self.mlp(Z_graph)
        out_age = self.mlp_age(Z_graph)
        
        # Node scores weighted by FC strength
        node_features = Z  # [batch_size, num_nodes, hidden_dim]
        node_scores_raw = self.node_score_mlp(node_features).squeeze(-1)  # [batch_size, num_nodes]
        node_score = node_scores_raw * fc_strength
        # node_score = node_scores_raw
        # import pdb;pdb.set_trace()
        contrastive_loss = contrastive_loss / out.shape[0]

        # import pdb;pdb.set_trace()
        return out, out_age, node_score, Z, contrastive_loss, fc_strength

    def update_fc_strength(self, fc_strength):
        """
        Update functional connectivity strength using normalization.
        """
        # Apply normalization
        fc_strength = F.relu(fc_strength)  # Ensure non-negative values
        fc_strength = fc_strength / (fc_strength.sum(dim=1, keepdim=True) + 1e-6)  # Normalize by sum
        return fc_strength