import jax
from ml_collections import ConfigDict


def make_config(z_dim=8, num_temps=4):
	cfg = ConfigDict()

	cfg.model = ConfigDict()
	cfg.model.z_dim = z_dim
	cfg.model.seed = 0
	cfg.model.base = "kaem"

	cfg.ebm = ConfigDict()
	cfg.ebm.layer_widths = 200, 200, 1
	cfg.ebm.leakyrelu_leak = 0.1
	cfg.ebm.p0_stddev = 1.0
	cfg.ebm.mcmc_stepsize = 0.01
	cfg.ebm.mcmc_numsteps = 5
	cfg.ebm.lr_init = 0.00002
	cfg.ebm.lr_end = 0.00001
	cfg.ebm.lr_decay = 0.975
	cfg.ebm.lr_beta1 = 0.999
	cfg.ebm.lr_beta2 = 0.5

	cfg.gen = ConfigDict()
	cfg.gen.cnn_channels = 16, 8, 4
	cfg.gen.kernel_sizes = (
		(4, 4),
		(4, 4),
		(4, 4),
	)
	cfg.gen.strides = (
		(1, 1),
		(2, 2),
		(2, 2),
	)
	cfg.gen.paddings = (
		"VALID",
		"SAME",
		"SAME",
	)
	cfg.gen.img_channels = 3
	cfg.gen.image_res = 32
	cfg.gen.leakyrelu_leak = 0.1
	cfg.gen.gaussian_stddev = 1.0
	cfg.gen.mcmc_stepsize = 0.01
	cfg.gen.mcmc_numsteps = 5
	cfg.gen.lr_init = 0.00002
	cfg.gen.lr_end = 0.00001
	cfg.gen.lr_decay = 0.975
	cfg.gen.lr_beta1 = 0.999
	cfg.gen.lr_beta2 = 0.5

	cfg.kan_prior = ConfigDict()
	cfg.kan_prior.basis = "rbf"
	cfg.kan_prior.kernel = "gaussian"
	cfg.kan_prior.mixture = True
	cfg.kan_prior.numquad = 25
	cfg.kan_prior.numgrid = 10

	cfg.thermo = ConfigDict()
	cfg.thermo.num_temps = num_temps
	cfg.thermo.xchange_every = 2

	cfg.training = ConfigDict()
	cfg.training.epochs = 1
	cfg.training.batch_size = 32
	cfg.training.dataset = "fake32"

	cfg.unbiased_metrics = ConfigDict()
	cfg.unbiased_metrics.batch_size_to_generate = 50
	cfg.unbiased_metrics.num_samples = 50
	cfg.unbiased_metrics.regression_steps = 1000, 1100, 1200

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
