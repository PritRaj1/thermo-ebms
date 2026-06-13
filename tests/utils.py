import jax
from ml_collections import ConfigDict


def make_config(z_dim=8, num_temps=4):
	cfg = ConfigDict()

	cfg.model = ConfigDict()
	cfg.model.z_dim = z_dim

	cfg.ebm = ConfigDict()
	cfg.ebm.energy_dim = 4
	cfg.ebm.leakyrelu_leak = 0.1
	cfg.ebm.p0_stddev = 1.0
	cfg.ebm.nuts_eta = 1e-1
	cfg.ebm.nuts_burn_in = 2
	cfg.ebm.nuts_numsteps = 5
	cfg.ebm.lr_init = 0.00002
	cfg.ebm.lr_end = 0.00001
	cfg.ebm.lr_decay = 0.975
	cfg.ebm.lr_beta1 = 0.999
	cfg.ebm.lr_beta2 = 0.5

	cfg.gen = ConfigDict()
	cfg.gen.hidden_dim = 8
	cfg.gen.img_channels = 3
	cfg.gen.image_res = 32
	cfg.gen.leakyrelu_leak = 0.1
	cfg.gen.gaussian_stddev = 1.0
	cfg.gen.nuts_eta = 1e-3
	cfg.gen.nuts_burn_in = 2
	cfg.gen.nuts_numsteps = 5
	cfg.gen.lr_init = 0.00002
	cfg.gen.lr_end = 0.00001
	cfg.gen.lr_decay = 0.975
	cfg.gen.lr_beta1 = 0.999
	cfg.gen.lr_beta2 = 0.5

	cfg.thermo = ConfigDict()
	cfg.thermo.num_temps = num_temps
	cfg.thermo.xchange_every = 2

	cfg.training = ConfigDict()
	cfg.training.epochs = 1
	cfg.training.batch_size = 128
	cfg.training.dataset = "cifar10"

	cfg.unbiased_metrics = ConfigDict()
	cfg.unbiased_metrics.batch_size_to_generate = 100
	cfg.unbiased_metrics.num_samples = 200

	cfg.logging = ConfigDict()
	cfg.logging.logdir = "/tmp/"
	cfg.logging.ckpt_dir = "/tmp/"
	cfg.logging.ckpt_every = 1
	cfg.logging.eval_every = 1
	cfg.logging.sample_every = 1
	cfg.logging.num_samples = 100

	cfg.lr_schedule = ConfigDict()
	cfg.lr_schedule.begin = 1
	cfg.lr_schedule.step = 1

	return cfg


def make_x(batch=4):
	return jax.random.normal(jax.random.key(0), (batch, 32, 32, 3))
