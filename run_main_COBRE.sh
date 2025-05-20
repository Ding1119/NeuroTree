data_type=cobre
num_epochs=300
batch_size=4
num_timesteps=2
num_nodes=118
input_dim=75
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