import jax
import jax.numpy as jnp
import configparser
import optax

from src.MCMC_Samplers.sample_distributions import sample_p0
from src.models.PriorModel import EBM
from src.models.GeneratorModel import GEN

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

E_lr_start = float(parser["OPTIMIZER"]["E_INITIAL_LR"])
G_lr_start = float(parser["OPTIMIZER"]["G_INITIAL_LR"])
E_lr_end = float(parser["OPTIMIZER"]["E_FINAL_LR"])
G_lr_end = float(parser["OPTIMIZER"]["G_FINAL_LR"])
gamma = float(parser["LR_SCHEDULE"]["DECAY_RATE"])
begin = int(parser["LR_SCHEDULE"]["BEGIN_EPOCH"]) * int(parser["PIPELINE"]["NUM_TRAIN_DATA"])/int(parser["PIPELINE"]["BATCH_SIZE"])
step = int(parser["LR_SCHEDULE"]["STEP_INTERVAL"]) * int(parser["PIPELINE"]["NUM_TRAIN_DATA"])/int(parser["PIPELINE"]["BATCH_SIZE"])

E_beta_1 = float(parser["OPTIMIZER"]["E_BETA_1"])
G_beta_1 = float(parser["OPTIMIZER"]["G_BETA_1"])
E_beta_2 = float(parser["OPTIMIZER"]["E_BETA_2"])
G_beta_2 = float(parser["OPTIMIZER"]["G_BETA_2"])


def init_EBM(key):
    """Initialise the EBM model and its parameters."""

    key, z_init = sample_p0(key)
    EBM_model = EBM()
    EBM_params = jax.jit(EBM_model.init)(key, z_init)
    EBM_fwd = jax.jit(EBM_model.apply)

    return key, EBM_params, EBM_fwd


def init_GEN(key):
    """Initialise the GEN model and its parameters."""

    key, z_init = sample_p0(key)
    GEN_model = GEN()
    GEN_params = jax.jit(GEN_model.init)(key, z_init)
    GEN_fwd = jax.jit(GEN_model.apply)

    return key, GEN_params, GEN_fwd


def init_EBM_optimiser(EBM_params):
    """Initialise the EBM optimiser and its state."""

    LR_schedule = optax.exponential_decay(init_value=E_lr_start, transition_steps=step, decay_rate=gamma, transition_begin=begin, end_value=E_lr_end)
    E_optimiser = optax.adam(LR_schedule, b1=E_beta_1, b2=E_beta_2)
    E_opt_state = E_optimiser.init(EBM_params)

    return E_optimiser, E_opt_state


def init_GEN_optimiser(GEN_params):
    """Initialise the GEN optimiser and its state."""

    LR_schedule = optax.exponential_decay(init_value=G_lr_start, transition_steps=step, decay_rate=gamma, transition_begin=begin, end_value=G_lr_end)
    GEN_optimiser = optax.adam(LR_schedule, b1=G_beta_1, b2=G_beta_2)
    GEN_opt_state = GEN_optimiser.init(GEN_params)

    return GEN_optimiser, GEN_opt_state
