data_type=cannabis
brain_tree_plot=False
num_epochs=300
batch_size=4
num_timesteps=2
num_nodes=90
input_dim=405
hidden_dim=64
num_classes=1


python main.py \
  --data_type ${data_type} \
  --brain_tree_plot ${brain_tree_plot} \
  --num_epochs ${num_epochs} \
  --batch_size ${batch_size} \
  --num_timesteps ${num_timesteps} \
  --num_nodes ${num_nodes} \
  --input_dim ${input_dim} \
  --hidden_dim ${hidden_dim} \
  --num_classes ${num_classes}