from execute_fixed_param import *

model_name = "ToggleSwitch"
labels = ["P1", "P2"]
colors = ["b", "r"]

noise_dim = 960
state_dim = 2
traj_len = 32

n_epochs = 1
batch_size = 256
n_critic = 5
n_gen = 1


do_train_flag = False

if do_train_flag:
	execute(model_name, state_dim, traj_len, 
			labels, colors, noise_dim = noise_dim, n_epochs = n_epochs, 
			batch_size = batch_size, n_critic = n_critic, n_gen = n_gen)
else:
	trained_model = "/final_generator_400_epochs.h5"
	model_id = "TACAS"
	evaluate_trained_model(trained_model, model_id, model_name, state_dim, traj_len, 
			labels, colors, noise_dim = noise_dim, n_epochs = n_epochs, 
			batch_size = batch_size, n_critic = n_critic, n_gen = n_gen)

	
