from WGAN_MA import *

def execute(model_name, state_dim, traj_len, 
			labels, colors, noise_dim = 480, n_epochs = 200, 
			batch_size = 256, n_critic = 1, n_gen = 1, 
			trainset_file = "_training_set.pickle",
			 valset_file = "_validation_set.pickle"):

	# Instantiate the c-WCGAN class
	wgan = WGAN_MA(model_name, noise_dim, state_dim, traj_len, labels, colors)
	wgan.generate_directories()

	# Load training and test data
	wgan.set_dataset_location(trainset_file, valset_file)
	wgan.load_real_data()
	wgan.load_test_data()

	# Set and train the model
	wgan.set_training_options(n_epochs, batch_size, n_critic, n_gen)
	wgan.define_wgan_model()
	wgan.print_log()
	wgan.train()

	# Evaluate the generator performances
	wgan.generate_validation_trajectories()
	wgan.plot_validation_trajectories()
	wgan.compute_distances()
	wgan.plot_last_step_histogram()


def evaluate_trained_model(trained_gen_file, model_id, model_name, state_dim, traj_len, 
			labels, colors, noise_dim = 480, n_epochs = 200, 
			batch_size = 256, n_critic = 1, n_gen = 1, 
			trainset_file = "_training_set.pickle",
			 valset_file = "_validation_set.pickle"):
	
	wgan = WGAN_MA(model_name, noise_dim, state_dim, traj_len, labels, colors)
	wgan.generate_directories(model_id)

	# Load training and test data
	wgan.set_dataset_location(trainset_file, valset_file)
	wgan.load_real_data()
	wgan.load_test_data()

	# Set and train the model
	wgan.set_training_options(n_epochs, batch_size, n_critic, n_gen)
	wgan.define_wgan_model()
	wgan.print_log()
	wgan.generator = load_model(wgan.MODELS_PATH+trained_gen_file,custom_objects={'Conv1DTranspose': Conv1DTranspose})

	# Evaluate the generator performances
	wgan.generate_validation_trajectories()
	wgan.plot_validation_trajectories()
	wgan.compute_distances()
	wgan.plot_last_step_histogram()