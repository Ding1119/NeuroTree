import numpy as np
import pandas as pd
from models.ode_model import *
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, f1_score, precision_score, recall_score, confusion_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from torch.utils.tensorboard import SummaryWriter
from tree_trunk_utils import *
from visualization.tree_plot import *


class NeuroODETrainer:
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train_step(self, A_s, A_d_seq, X_seq, labels, age):
        self.model.train()
        self.optimizer.zero_grad()
        outputs, outputs_age, node_scores, _, cl_loss, fc_strength = self.model(A_s, A_d_seq, X_seq, age)
        if self.model.num_classes == 1:
            loss = F.mse_loss(outputs_age.squeeze(), age)
        else:
            loss = F.cross_entropy(outputs, labels) + cl_loss
            # loss = F.cross_entropy(outputs, labels)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, A_s, A_d_seq, X_seq, labels, age):
        self.model.eval()
        outputs, outputs_age, node_scores  ,_, cl_loss, fc_strength = self.model(A_s, A_d_seq, X_seq, age)
        if self.model.num_classes == 1:
            loss = F.mse_loss(outputs_age.squeeze(), age)
            predictions = outputs_age
        else:
            loss = F.cross_entropy(outputs, labels) +  cl_loss
            # loss = F.cross_entropy(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
        return predictions, loss.item()


def create_neuro_ode_model(num_nodes, input_dim, hidden_dim, num_classes, num_timesteps):
    model = NeuroODE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_timesteps=num_timesteps,
        num_nodes=num_nodes)
    
    return model

def train_model(model, train_loader, test_loader, num_epochs, device):
    trainer = NeuroODETrainer(model)
    model = model.to(device)
    best_test_loss = float('inf')
    patience_counter = 0
    max_patience = 100
    for epoch in range(num_epochs):
        train_losses = []
        model.train()
        for A_s, A_d_seq, X_seq, labels, age, _ in train_loader:
            A_s = A_s.to(device)
            A_d_seq = A_d_seq.to(device)
            X_seq = X_seq.to(device)
            labels = labels.to(device)
            age = age.to(device)
            loss = trainer.train_step(A_s, A_d_seq, X_seq, labels, age)
            loss = loss / len(A_s)
            train_losses.append(loss)

        test_losses = []
        model.eval()
        with torch.no_grad():
            for A_s, A_d_seq, X_seq, labels, age, _ in test_loader:
                A_s = A_s.to(device)
                A_d_seq = A_d_seq.to(device)
                X_seq = X_seq.to(device)
                labels = labels.to(device)
                age = age.to(device)
                _, loss = trainer.evaluate(A_s, A_d_seq, X_seq, labels, age)
                loss = loss / len(A_s)
                test_losses.append(loss)

        avg_train_loss = np.mean(train_losses)
        avg_test_loss = np.mean(test_losses)

        # if avg_test_loss < best_test_loss:
        #     best_test_loss = avg_test_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1

        # if patience_counter >= max_patience:
        #     print(f"Early stopping at epoch {epoch}")
        #     break

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, test Loss = {avg_test_loss:.4f}")


def evaluate_final_model(model, test_loader, device, save_plots=True, plot_limit=None, data_type=None, brain_tree_plot=False):

    model.eval()
    all_labels = []
    all_predictions = []
    all_outputs = []
    all_ages = []
    all_output_ages = []
    all_trunk_list = []
    all_new_edge_lists = []

    if data_type == 'cannabis':
        df_roi = pd.read_csv('./datasets/cannabis/data/Cannabis_stanford_network_mapping.csv')
        network_mapping = dict(zip(df_roi['parcel_ind'].astype(int), df_roi['yeo_network']))
        coords = np.array([eval(coord) for coord in np.array(df_roi['coordinates'])])

    elif data_type == 'cobre':
        df_roi = pd.read_csv('./datasets/COBRE/COBRE_harvard_oxford_network_mapping_updated.csv')
        network_mapping = dict(zip(df_roi['parcel_ind'].astype(int), df_roi['yeo_network']))

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


    with torch.no_grad():
        for A_s, A_d_seq, X_seq, labels, age, edge_lists in test_loader:
            A_s = A_s.to(device)
            A_d_seq = A_d_seq.to(device)
            X_seq = X_seq.to(device)
            labels = labels.to(device)
            age = age.to(device)
            
            outputs, output_age, node_scores, Z, cl_loss, fc_strength  = model(A_s, A_d_seq, X_seq, age)
            predictions = torch.argmax(outputs, dim=1)
            # print(labels)

            edge_lists_batch = edge_lists  
            edge_lists_batch_Z = []
            for i in range(Z.size(0)):
                Z_sample = Z[i].cpu().numpy()
                FC = np.matmul(Z_sample, Z_sample.T)    
                # import pdb;pdb.set_trace()
                spanning_tree = minimum_spanning_tree(FC)
                
                brain_tree = (spanning_tree + spanning_tree.T) > 0
                brain_tree = brain_tree.toarray().astype(float)

                row, col = brain_tree.nonzero()
                edge_list = list(zip(row, col))
                edge_lists_batch_Z.append(edge_list)
  
            node_scores_batch = node_scores.cpu()
            # import pdb;pdb.set_trace()
            # trunks_batch = find_trunk_gnn_batch(node_scores_batch, edge_lists_batch, max_level=3)
   
            trunks_batch = find_trunk_gnn_batch_with_fc_strength(
            node_scores_batch=node_scores.cpu(),
            edge_lists_batch=edge_lists_batch,
            fc_strength_batch=fc_strength.cpu(),
            max_level=3
                )
            
            labels_np = labels.cpu().numpy()
            for i, (trunk, label) in enumerate(zip(trunks_batch, labels_np)):
                for level, path in enumerate(trunk, 1):
                    for node_idx in path:
                        if isinstance(node_idx, (list, tuple)):
                            node_idx = node_idx[0]
                        node_idx = int(node_idx)
                        network = network_mapping.get(node_idx, 'Others')
                        network_stats[label][f'level{level}'][network] += 1
  
            ########## Plot Brain Tree #########################
            if brain_tree_plot == True:
                plot_brain_tree(
                    data_type=data_type,
                    trunks_batch=trunks_batch,
                    # edge_lists_batch=edge_lists_batch_Z,
                    edge_lists_batch=edge_lists_batch,
                    df_roi=df_roi,
                    plot_dir='trunk_plots',
                    save_plots=save_plots,
                    plot_limit=plot_limit
                )

            ###########################
        
            all_trunk_list.extend(trunks_batch)            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            all_ages.extend(age.cpu().numpy())
            all_output_ages.extend(output_age.cpu().numpy())
            all_new_edge_lists.extend(edge_lists)

    if brain_tree_plot == True:   
        nature_colors = get_nature_colors()
        for label in [0, 1]:
            fig = create_combined_network_plot(network_stats, label, nature_colors)
            plt.savefig(f'network_distribution_label_{label}_{data_type}.png', 
                    bbox_inches='tight', dpi=300)
            plt.close() 


    accuracy = accuracy_score(all_labels, all_predictions)
    if model.num_classes == 2:

        probs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()
        auc = roc_auc_score(all_labels, probs[:, 1]) 
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        cm = confusion_matrix(all_labels, all_predictions)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)

        print(f"Final Model Binary Classification Metrics:")
        print(f"Accuracy = {accuracy:.4f}")
        print(f"Precision = {precision:.4f}")
        print(f"Recall/Sensitivity = {recall:.4f}")
        print(f"Specificity = {specificity:.4f}")
        print(f"F1 Score = {f1:.4f}")
        print(f"AUC-ROC = {auc:.4f}")
    else:
        auc = None


    mse_age = mean_squared_error(all_ages, all_output_ages)
    
    if auc is not None:
        print(f"Final Model Prediction: Accuracy = {accuracy:.4f}, AUC = {auc:.4f}")
    else:
        print(f"Final Model Prediction: Accuracy = {accuracy:.4f}, MSE (Age) = {mse_age:.4f}")
    
    predictions_array = np.argmax(all_outputs, axis=1)
    print("Predictions:", predictions_array.flatten())

    return all_trunk_list
