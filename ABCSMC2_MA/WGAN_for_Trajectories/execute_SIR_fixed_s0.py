from execute_fixed_s0 import *

model_name = "SIR_fixed_s0"
labels = ["S", "I"]
colors = ["b", "r"]

noise_dim = 480
state_dim = 2
param_dim = 2
traj_len = 16

n_epochs = 200
batch_size = 256
n_critic = 5
n_gen = 1

embedding = "REPEAT"#"DENSE"

trainset_file = "_training_set_dt=5.pickle"
valset_file = "_validation_set_dt=5.pickle"

do_train_flag = True

if do_train_flag:
	model_id = None
	execute(model_name, model_id, state_dim, param_dim, traj_len, 
			labels, colors, noise_dim = noise_dim, n_epochs = n_epochs, 
			batch_size = batch_size, n_critic = n_critic, n_gen = n_gen, 
			trainset_file = trainset_file, valset_file = valset_file)#, embedding = embedding)
else:
	trained_model = "/final_generator_500_epochs.h5"
	model_id = "TACAS"
	evaluate_trained_model(trained_model, model_id, model_name, state_dim, param_dim, traj_len, 
			labels, colors, noise_dim = noise_dim, n_epochs = n_epochs, 
			batch_size = batch_size, n_critic = n_critic, n_gen = n_gen)#, embedding = embedding)

	