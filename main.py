import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from data_handler.load_dataset import *
from training_eval_utils import *
from sklearn.model_selection import train_test_split



def train_eval(data_type, num_epochs, batch_size, num_nodes, num_timesteps, input_dim, hidden_dim, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"=======Loading {data_type} dataset...=======")

    if data_type == 'cannabis':
        A_s, A_f_seq, X_seq, labels, ages, edge_lists = dataloader(data_type, num_timesteps)
        

    elif data_type == 'cobre':
        A_s, A_f_seq, X_seq, labels, ages, edge_lists = dataloader(data_type, num_timesteps)

    num_samples = len(A_s)
    indices = np.arange(num_samples)
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_edge_lists = [edge_lists[i] for i in train_indices]
    test_edge_lists = [edge_lists[i] for i in test_indices]

    train_dataset = BrainNetworkDataset(
                        A_s[train_indices],
                        A_f_seq[train_indices],
                        X_seq[train_indices],
                        labels[train_indices],
                        ages[train_indices],
                        train_edge_lists)

    test_dataset = BrainNetworkDataset(
                        A_s[test_indices],
                        A_f_seq[test_indices],
                        X_seq[test_indices],
                        labels[test_indices],
                        ages[test_indices],
                        test_edge_lists)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate)

    print("Creating model...")
    model = create_ste_ode_model(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_timesteps=num_timesteps)

    print("Training model...")
    print(f"Using device: {device}")
    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        device=device)
    
    print("Evaluating final model...")

    trunks = evaluate_final_model(model, test_loader, device, save_plots=True, plot_limit=10, data_type=data_type)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='cannabis')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_nodes', type=int, default=90)
    parser.add_argument('--num_timesteps', type=int, default=2)
    parser.add_argument('--input_dim', type=int, default=405)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()

    train_eval(args.data_type, args.num_epochs, args.batch_size, args.num_nodes, args.num_timesteps,
               args.input_dim, args.hidden_dim, args.num_classes)
    

if __name__ == '__main__':
    main()
