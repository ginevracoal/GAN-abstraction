from execute import *

model_name = "eSIRS_1P"
labels = ["S", "I"]
colors = ["b", "r"]

noise_dim = 240
state_dim = 2
param_dim = 1
traj_len = 32

n_epochs = 1#200
batch_size = 1024#256
n_critic = 10
n_gen = 1


do_train_flag = False

if do_train_flag:
	execute(model_name, state_dim, param_dim, traj_len, 
			labels, colors, noise_dim = noise_dim, n_epochs = n_epochs, 
			batch_size = batch_size, n_critic = n_critic, n_gen = n_gen)
else:
	trained_model = "/final_generator_200_epochs.h5"
	model_id = "TACAS"
	evaluate_trained_model(trained_model, model_id, model_name, state_dim, param_dim, traj_len, 
			labels, colors, noise_dim = noise_dim, n_epochs = n_epochs, 
			batch_size = batch_size, n_critic = n_critic, n_gen = n_gen)

	