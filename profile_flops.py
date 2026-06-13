import jax
import matplotlib.pyplot as plt
from matplotlib import rc
from functools import partial
import pandas as pd
import numpy as np
import configparser

from src.pipeline.initialise import *
from src.pipeline.pipeline_steps import val_step

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

num_temps = int(parser["TEMP"]["NUM_TEMPS"])
batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
posterior_mcmc = int(parser["MCMC"]["G_SAMPLE_STEPS"])

"""Get number of flops to do validation"""

# Initialise the pipeline
key = jax.random.PRNGKey(0)
key, EBM_params, EBM_fwd = init_EBM(key)
key, GEN_params, GEN_fwd = init_GEN(key)
EBM_optimiser, EBM_opt_state = init_EBM_optimiser(EBM_params)
GEN_optimiser, GEN_opt_state = init_GEN_optimiser(GEN_params)

# Tuple up for cleanliness
params_tup = (EBM_params, GEN_params)
fwd_fcn_tup = (EBM_fwd, GEN_fwd)

test_x = np.random.randn(batch_size, 64, 64, 1)

# Get the FLOPs estimate from the cost analysis
wrapped = jax.xla_computation(partial(val_step, fwd_fcn_tup=fwd_fcn_tup))
computation = wrapped(key, test_x, params_tup)

print(repr(computation))
module = computation.as_hlo_module()
print(repr(module))
client = jax.lib.xla_bridge.get_backend()
print(repr(client))
analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, module)
for key, value in analysis.items():
	print("{}: {}".format(key, value))

# Save the results
df = pd.DataFrame(
	{
		"num_temps": [num_temps],
		"posterior_mcmc": [posterior_mcmc],
		"flops": [analysis["flops"]],
	}
)
df.to_csv("results/flops.csv", mode="a", header=False, index=False)
