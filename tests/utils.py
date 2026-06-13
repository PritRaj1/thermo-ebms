import jax
from ml_collections import ConfigDict


def make_config(z_dim=8, num_temps=5):
	cfg = ConfigDict()

	cfg.model = ConfigDict()
	cfg.model.z_dim = z_dim

	cfg.ebm = ConfigDict()
	cfg.ebm.energy_dim = 32
	cfg.ebm.leakyrelu_leak = 0.1
	cfg.ebm.p0_stddev = 1.0
	cfg.ebm.nuts_eta = 1e-1
	cfg.ebm.nuts_burn_in = 20
	cfg.ebm.nuts_numsteps = 1000

	cfg.gen = ConfigDict()
	cfg.gen.hidden_dim = 32
	cfg.gen.img_channels = 3
	cfg.gen.image_res = 32
	cfg.gen.leakyrelu_leak = 0.1
	cfg.gen.gaussian_stddev = 1.0
	cfg.gen.nuts_eta = 1e-3
	cfg.gen.nuts_burn_in = 10
	cfg.gen.nuts_numsteps = 100

	cfg.thermo = ConfigDict()
	cfg.thermo.num_temps = num_temps
	cfg.thermo.annealing_cycles = 1.0
	cfg.thermo.p_start = 0.5
	cfg.thermo.p_end = 2.0

	cfg.training = ConfigDict()
	cfg.training.epochs = 100
	cfg.training.numdata = 1000
	cfg.training.batch_size = 100

	return cfg


def make_x(batch=4):
	return jax.random.normal(jax.random.key(0), (batch, 32, 32, 3))
