from execute import *

model_name = "SIR"
labels = ["S", "I", "R"]
colors = ["b", "r", "g"]

noise_dim = 480
state_dim = 3
traj_len = 16

n_epochs = 1
batch_size = 256
n_critic = 2
n_gen = 1


do_train_flag = False

if do_train_flag:
	execute(model_name, state_dim, traj_len, 
			labels, colors, noise_dim = noise_dim, n_epochs = n_epochs, 
			batch_size = batch_size, n_critic = n_critic, n_gen = n_gen)
else:
	trained_model = "/final_generator_500_epochs.h5"
	model_id = "TACAS"
	evaluate_trained_model(trained_model, model_id, model_name, state_dim, traj_len, 
			labels, colors, noise_dim = noise_dim, n_epochs = n_epochs, 
			batch_size = batch_size, n_critic = n_critic, n_gen = n_gen)

	