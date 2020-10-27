# GAN-abstraction

### Abstraction of Markov Population Dynamics via Generative Adversarial Nets

Luca Bortolussi, Francesca Cairoli, Ginevra Carbone.

The code containing the experiments and the results presented at the TACAS 21 conference are located in the `TACAS_21` folder.
The folder is organized as follows:
- `Dataset_Generation` folder that contains the scripts to generate the datasets through SSA simulation (based on Stochpy library)
- `WGAN_for_Trajectories` folder that contains the implementation of teh c-WCGAN with either fixed parameters (`class WGAN_MA_fixed_param`) or with an arbitrary number of varying parameters (`class WGAN_MA`).

The datasets used are shared at the following link: https://www.dropbox.com/home/Datasets%20for%20GAN-abstraction


