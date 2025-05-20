data_type=cannabis
num_epochs=100
batch_size=16
num_timesteps=2
num_nodes=90
input_dim=405
hidden_dim=64
num_classes=2


python main.py \
  --data_type ${data_type} \
  --num_epochs ${num_epochs} \
  --batch_size ${batch_size} \
  --num_timesteps ${num_timesteps} \
  --num_nodes ${num_nodes} \
  --input_dim ${input_dim} \
  --hidden_dim ${hidden_dim} \
  --num_classes ${num_classes}